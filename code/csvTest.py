import pandas as pd

submission = pd.read_csv(
    "/data/zhaoxinying/code/rsna-miccai-2021/user_data/preds/submission.csv",
    index_col="BraTS21ID")

ids = [1, 13]
print(ids)
y_pred = [0.8, 0.6]
preddf = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": y_pred})
preddf = preddf.set_index("BraTS21ID")

submission["MGMT_value"] = 0
submission["MGMT_value"] += preddf["MGMT_value"]
# print(submission["MGMT_value"])

print(submission["MGMT_value"])
submission["MGMT_value"].to_csv("/data/zhaoxinying/code/rsna-miccai-2021/user_data/preds/efficientnet3d_b0_lr0.0003_aug256_2/submission_test.csv")