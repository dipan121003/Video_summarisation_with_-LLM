import os
import glob
import gc
import torch
import numpy as np
import cv2
import scipy.io
import h5py
import whisper
from scipy.io import loadmat
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from ultralytics import YOLO
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TVSUM_VIDEO_DIR = "/DATA/CV/datasets/ydata-tvsum50-v1_1/video/*.mp4"
TVSUM_MAT_PATH = "/DATA/CV/datasets/ydata-tvsum50-v1_1/matlab/ydata-tvsum50.mat"
SAVE_TVSUM = "./processed_data_2FPS/tvsum"

SUMME_VIDEO_DIR = "/DATA/CV/datasets/SumMe/videos/*.mp4"
SUMME_GT_DIR = "/DATA/CV/datasets/SumMe" 
SAVE_SUMME = "./processed_data_2FPS/summe"

LLAVA_ID = "llava-hf/llava-1.5-7b-hf"
LLAMA_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# ==========================================
# GT LOADING UTILS
# ==========================================
def load_summe_scores(mat_folder, video_name):
    name = os.path.splitext(video_name)[0]
    path = os.path.join(mat_folder, "GT", f"{name}.mat")
    if not os.path.exists(path): path = os.path.join(mat_folder, f"{name}.mat")
    if not os.path.exists(path): return None
    try:
        mat = loadmat(path)
        scores = mat["user_score"]
        avg = scores.mean(axis=1)
        return (avg - avg.min()) / (avg.max() - avg.min() + 1e-6)
    except: return None

def load_tvsum_scores(mat_path, video_name):
    name = os.path.splitext(video_name)[0]
    try:
        with h5py.File(mat_path, "r") as f:
            tvsum = f["tvsum50"]
            # Map filenames to keys safely
            vid_refs = tvsum["video"][:]
            names = []
            for ref in vid_refs:
                ref_obj = f[ref[0]] if isinstance(ref, np.ndarray) else f[ref]
                names.append(''.join(chr(c) for c in ref_obj[()].squeeze()))
            
            if name not in names: return None
            idx = names.index(name)
            
            gt_ref = tvsum["gt_score"][idx][0]
            scores = f[gt_ref][()].squeeze()
            return (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    except: return None

# ==========================================
# VIDEO-WISE PROCESSOR
# ==========================================
class VideoProcessor:
    def __init__(self):
        self.yolo = YOLO("yolov8n.pt")
        
    def cleanup(self):
        torch.cuda.empty_cache()
        gc.collect()

    # -------------------------------------------------------
    # STEP 1: GENERATE TEXT DESCRIPTIONS (Audio + Vision)
    # -------------------------------------------------------
    def get_multimodal_prompts(self, video_path, use_audio=True, sample_fps=2):
        print(f"   [1/2] Extracting Description (Audio + Vision)...")
        
        # --- A. AUDIO (WHISPER) ---
        transcript = []
        if use_audio:
            try:
                w_model = whisper.load_model("small").to(DEVICE)
                trans = w_model.transcribe(video_path)
                transcript = trans['segments']
                del w_model
                self.cleanup()
            except Exception as e:
                print(f"     Whisper Error: {e}")

        # --- B. VISION (LLaVA) ---
        processor = AutoProcessor.from_pretrained(LLAVA_ID)
        v_model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_ID, torch_dtype=torch.float16
        ).to(DEVICE)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(fps / sample_fps))
        
        prompts = []
        frames_batch = []
        times_batch = []
        
        # Process in chunks to speed up LLaVA
        batch_size = 8
        
        pbar = tqdm(total=total // step, desc="     Frames", leave=False)
        
        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break
            
            frames_batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            times_batch.append(i/fps)
            
            if len(frames_batch) >= batch_size:
                prompts.extend(self._process_vision_batch(v_model, processor, frames_batch, times_batch, transcript))
                frames_batch = []
                times_batch = []
                pbar.update(batch_size)
                
        # Leftovers
        if frames_batch:
            prompts.extend(self._process_vision_batch(v_model, processor, frames_batch, times_batch, transcript))
            pbar.update(len(frames_batch))
            
        pbar.close()
        cap.release()
        
        del v_model, processor
        self.cleanup()
        return prompts

    def _process_vision_batch(self, model, processor, frames, timestamps, transcript):
        # Prepare LLaVA Inputs
        txt = "USER: <image>\nDescribe the Location and Action concisely.\nASSISTANT:"
        inputs = processor(images=frames, text=[txt]*len(frames), return_tensors="pt", padding=True).to(DEVICE, torch.float16)
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=40)
        captions = processor.batch_decode(out, skip_special_tokens=True)
        
        batch_res = []
        for i, raw_cap in enumerate(captions):
            caption = raw_cap.split("ASSISTANT:")[-1].strip()
            
            # YOLO
            y_res = self.yolo(frames[i], verbose=False)[0]
            objs = list(set([self.yolo.names[int(c)] for c in y_res.boxes.cls]))
            obj_str = ", ".join(objs[:5]) if objs else "None"
            
            # Audio Sync
            aud_txt = ""
            if transcript:
                t = timestamps[i]
                for seg in transcript:
                    if seg['start'] <= t <= seg['end']:
                        aud_txt = f" Audio: '{seg['text']}'"
                        break
            
            batch_res.append(f"Visual: {caption}. Objects: {obj_str}.{aud_txt}")
        
        return batch_res

    # -------------------------------------------------------
    # STEP 2: GENERATE EMBEDDINGS (Llama 3 with Context)
    # -------------------------------------------------------
    def get_llama_embeddings(self, prompts, window_size=5):
        print(f"   [2/2] Embedding with Llama 3 (Window {window_size})...")
        
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_ID)
        tokenizer.pad_token = tokenizer.eos_token
        l_model = AutoModelForCausalLM.from_pretrained(LLAMA_ID, torch_dtype=torch.float16).to(DEVICE)
        
        half_win = window_size // 2
        embeddings = []
        
        # --- FEW SHOT EXAMPLES ---
        few_shot_msgs = [
            {"role": "system", "content": "You are a professional video editor. Analyze the 'TARGET FRAME' context to determine its importance."},
            {"role": "user", "content": (
                "[Previous]: Visual: Man typing. Objects: desk.\n"
                "[TARGET FRAME]: Visual: Man typing. Objects: desk.\n"
                "[Next]: Visual: Man scratching nose. Objects: desk."
            )},
            {"role": "assistant", "content": "Analysis: Low importance. Repetitive visual content with no significant action."},
            {"role": "user", "content": (
                "[Previous]: Visual: Soccer player running.\n"
                "[TARGET FRAME]: Visual: Player kicks ball into goal. Audio: Crowd cheers.\n"
                "[Next]: Visual: Players celebrating."
            )},
            {"role": "assistant", "content": "Analysis: High importance. Climax of the event with distinct visual action and audio intensity."}
        ]
        
        # Prepare Context Windows
        context_inputs = []
        n_frames = len(prompts)
        for i in range(n_frames):
            start = max(0, i - half_win)
            end = min(n_frames, i + half_win + 1)
            win_txts = prompts[start:end]
            
            ctx_str = ""
            for offset, txt in enumerate(win_txts):
                pos = start + offset - i
                label = "Previous" if pos < 0 else "Next" if pos > 0 else "TARGET FRAME"
                ctx_str += f"[{label}]: {txt}\n"
            context_inputs.append(ctx_str)
            
        # Batch Inference
        batch_size = 4
        for i in tqdm(range(0, len(context_inputs), batch_size), desc="     Embedding", leave=False):
            batch = context_inputs[i:i+batch_size]
            formatted = []
            for txt in batch:
                msgs = list(few_shot_msgs)
                msgs.append({"role": "user", "content": txt})
                formatted.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
                
            inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(DEVICE)
            with torch.no_grad():
                out = l_model(**inputs, output_hidden_states=True)
                
            hidden = out.hidden_states[-1]
            for idx in range(hidden.shape[0]):
                seq_len = inputs.attention_mask[idx].sum()
                emb = hidden[idx, seq_len-1, :].float().cpu().numpy()
                embeddings.append(emb)
                
        del l_model, tokenizer
        self.cleanup()
        return np.vstack(embeddings)

    # -------------------------------------------------------
    # MASTER LOOP
    # -------------------------------------------------------
    def process_dataset(self, vid_files, save_dir, gt_func, gt_src, use_audio):
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        print(f"Processing {len(vid_files)} videos to {save_dir}...")
        
        for vid_path in vid_files:
            name = os.path.basename(vid_path).split('.')[0]
            save_path = os.path.join(save_dir, f"{name}.pt")
            
            if os.path.exists(save_path):
                print(f"Skipping {name} (Done)")
                continue
                
            print(f"--> Processing: {name}")
            
            # Check GT first
            gt_scores = gt_func(gt_src, name)
            if gt_scores is None:
                print(f"    Skipping {name} (No GT found)")
                continue
            
            # 1. Get Prompts (Loads/Unloads Whisper & LLaVA)
            prompts = self.get_multimodal_prompts(vid_path, use_audio=use_audio)
            
            # 2. Get Embeddings (Loads/Unloads Llama)
            feats = self.get_llama_embeddings(prompts)
            
            # 3. Resize GT & Save
            gt_np = torch.tensor(gt_scores, dtype=torch.float32).numpy()
            gt_resized = cv2.resize(gt_np.reshape(1,-1), (feats.shape[0], 1)).flatten()
            
            torch.save({
                'features': torch.tensor(feats, dtype=torch.float16),
                'gt_scores': torch.tensor(gt_resized, dtype=torch.float32)
            }, save_path)
            print(f"    Saved {name}. Shape: {feats.shape}")

# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    proc = VideoProcessor()
    
    # TVSum (Audio Enabled)
    tv_vids = glob.glob(TVSUM_VIDEO_DIR)
    if tv_vids:
        proc.process_dataset(tv_vids, SAVE_TVSUM, load_tvsum_scores, TVSUM_MAT_PATH, use_audio=True)
        
    # SumMe (Audio Disabled)
    sm_vids = glob.glob(SUMME_VIDEO_DIR)
    if sm_vids:
        proc.process_dataset(sm_vids, SAVE_SUMME, load_summe_scores, SUMME_GT_DIR, use_audio=False)
