import torch

num_samples = 796*256
feature_dim = 384

try:
    delta = torch.load("all_diffs_tensor.pt")
    print(f"Original tensor loaded. Shape: {delta.shape}")
    non_zero_rows_mask = torch.any(delta != 0, dim=1)
    
    filtered_delta = delta[non_zero_rows_mask]

    num_remaining_rows = filtered_delta.shape[0]

    print(f"\nShape after removing all-zero rows: {filtered_delta.shape}")
    print(f"Final count of non-zero rows: {num_remaining_rows}")

except FileNotFoundError:
    print("Error: 'all_diffs_tensor.pt' 파일을 찾을 수 없습니다.")

# delta = torch.load("all_diffs_tensor.pt")
# print(f"수집된 delta F 텐서 크기: {delta.shape}")


# 2. PCA 수행
# torch.pca_lowrank는 데이터를 중앙 정렬(mean-centering)한 후 SVD를 수행합니다.
# V.T가 고유벡터(principal components)에 해당하며, 각 행이 하나의 고유벡터입니다.
# S는 각 고유벡터에 대응하는 특이값(singular values)입니다.
# 특이값은 분산의 제곱근에 비례하며, 내림차순으로 정렬됩니다.
try:
    U, S, V = torch.pca_lowrank(delta, q=feature_dim)
except RuntimeError as e:
    print(f"Error during PCA: {e}")
    print("Trying with q=num_samples")
    U, S, V = torch.pca_lowrank(delta, q=num_samples)


# 3. 변화에 가장 둔감한 벡터(V) 찾기
# S와 V는 분산이 큰 순서(내림차순)로 정렬되어 있습니다.
# 따라서 우리는 가장 마지막에 있는 고유벡터를 선택해야 합니다.
# 이 벡터가 가장 작은 분산을 가지는 방향, 즉 변화에 가장 둔감한 방향입니다.
# least_variant_eigenvector = V[-1, :]  # V의 마지막 행
most_variant_eigenvector = V[0, :]  # V의 마지막 행

# 4. 최종 벡터 형태 맞추기
# (384,) -> (384, 1) 로 변환하여 행렬 곱셈에 사용하기 용이하게 만듭니다.
V_final = most_variant_eigenvector.unsqueeze(0)  # (1, 384)

# torch.save(V_final, "insensitive_vec.pt")
torch.save(V_final, "sensitive_vec.pt")

print("\nPCA 결과 V의 크기 (고유벡터 행렬):", V.shape)
print("가장 작은 고유값에 해당하는 고유벡터(V_final) 크기:", V_final.shape)
print("\n최종적으로 찾은 predefined vector V_final:")
print(V_final)