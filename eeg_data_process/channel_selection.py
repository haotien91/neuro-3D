# eeg_data_process/channel_selection.py

"""
Channel selection for simulating 32-channel easyCap layout.

Mapping strategy for missing channels:
- PO5 → PO7, PO7 → PO9 (left hemisphere shift)
- PO6 → PO8, PO8 → PO10 (right hemisphere shift)
- AFz → FPZ (closest midline frontal)
"""

# 標準 32-channel BrainCap (BC-32) 目標配置（依 PDF 表列順序，移除 IO 空白座標）
# 原 PDF 列：Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz, IO, FC1, FC2, CP1,
#           CP2, FC5, FC6, CP5, CP6, FT9, FT10, TP9, TP10
# 注意：資料中的 64ch 排列沒有 FT9/FT10/TP9/TP10，且沒有 IO；我們在替代規則中處理。
BC32_TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz',
    'IO',  # 由替代規則映射為 'OZ'
    'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10'
]

BC22_TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'IO', 'TP9', 'TP10'
]

# 64-channel data 的實際順序（來自 GitHub Issue #2）
EASYCAP_64_CHANNELS = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
    'C6', 'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 
    'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 
    'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1',
    'O1', 'OZ', 'O2', 'CB2'
]

def get_channel_indices():
    """
    Map 32-channel easyCap to 64-channel data with substitutions.
    """
    indices = []
    found = []
    substitutions_used = []
    
    channels_64_upper = [ch.upper() for ch in EASYCAP_64_CHANNELS]
    
    # Substitution mapping
    substitutions = {
        # BC-32/64 PDF 中的名稱替代 → 以資料中實際存在之通道名稱替代
        # 無 IO：使用最接近的正中後部 Oz
        'IO': 'OZ',
        # 無 FT9/FT10：以 FT7/FT8 替代（顳額部鄰近）
        'FT9': 'FT7',
        'FT10': 'FT8',
        # 無 TP9/TP10：以 TP7/TP8 替代（顳頂部鄰近）
        'TP9': 'TP7',
        'TP10': 'TP8',
        # 若 BC-32 之中出現 Fpz/AFz/FCz 等，資料中對應為 FPZ/FCZ（現列表未包含，但保留以防擴充）
        'AFZ': 'FPZ',
        'FPZ': 'FPZ',
    }
    
    for ch in BC32_TARGET_CHANNELS:
        ch_upper = ch.upper()
        
        # Check if substitution needed
        if ch_upper in substitutions:
            target = substitutions[ch_upper]
            substitutions_used.append(f"{ch}→{target}")
            ch_upper = target
        
        if ch_upper in channels_64_upper:
            idx = channels_64_upper.index(ch_upper)
            indices.append(idx)
            found.append(ch)
        else:
            # 若仍找不到，記錄缺失
            substitutions_used.append(f"{ch}→(missing)")
    
    # 去重，並驗證數量
    unique_indices = []
    seen = set()
    for i in indices:
        if i not in seen:
            seen.add(i)
            unique_indices.append(i)

    if len(unique_indices) != 32:
        print(f"[BC-32 Mapping Warning] Expected 32 channels but got {len(unique_indices)}. Indices: {unique_indices}")
        # 若多於/少於 32，可在此加上更嚴格處理；目前直接斷言避免靜默錯誤
        raise ValueError(f"BC-32 mapping produced {len(unique_indices)} channels; expected 32")

    print(f"[32-Channel Selection]")
    print(f"  Target: {len(BC32_TARGET_CHANNELS)} channels")
    print(f"  Found: {len(found)} channels")
    print(f"  Substitutions: {', '.join(substitutions_used)}")
    
    return unique_indices

CHANNEL_INDICES_64_TO_32 = get_channel_indices()

def get_channel_indices_22():
    indices = []
    found = []
    substitutions_used = []
    channels_64_upper = [ch.upper() for ch in EASYCAP_64_CHANNELS]
    substitutions = {
        'IO': 'OZ',
        'TP9': 'TP7',
        'TP10': 'TP8',
    }
    for ch in BC22_TARGET_CHANNELS:
        ch_upper = ch.upper()
        if ch_upper in substitutions:
            target = substitutions[ch_upper]
            substitutions_used.append(f"{ch}→{target}")
            ch_upper = target
        if ch_upper in channels_64_upper:
            idx = channels_64_upper.index(ch_upper)
            indices.append(idx)
            found.append(ch)
        else:
            substitutions_used.append(f"{ch}→(missing)")
    unique_indices = []
    seen = set()
    for i in indices:
        if i not in seen:
            seen.add(i)
            unique_indices.append(i)
    if len(unique_indices) != 22:
        print(f"[BC-22 Mapping Warning] Expected 22 channels but got {len(unique_indices)}. Indices: {unique_indices}")
        raise ValueError(f"BC-22 mapping produced {len(unique_indices)} channels; expected 22")
    print(f"[22-Channel Selection]")
    print(f"  Target: {len(BC22_TARGET_CHANNELS)} channels")
    print(f"  Found: {len(found)} channels")
    print(f"  Substitutions: {', '.join(substitutions_used)}")
    return unique_indices

CHANNEL_INDICES_64_TO_22 = get_channel_indices_22()

if __name__ == '__main__':
    print(f"\nSelected indices: {CHANNEL_INDICES_64_TO_32}")
    print(f"Total: {len(CHANNEL_INDICES_64_TO_32)} channels")