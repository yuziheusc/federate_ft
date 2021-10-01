from federate_ft.fl_main import SyncFl

import glob
import pickle

if __name__ == "__main__":
    
    path = "./data_syn"
    res_path = "./res_data_syn.pkl"
    splits = sorted(glob.glob(f"{path}/*"))
    print(splits)

    res_list_raw = []
    res_list_ft = []
    for split in splits:
        
        fl_model = SyncFl(split, x_dim=3, nz=2)
        fl_model.train(16, 4)
        fl_model.train_ft(16, 4)
        
        res_raw = fl_model.test(fl_model.make_loader_ft(f"{split}/test.npz"))
        res_ft = fl_model.test_ft(fl_model.make_loader_ft(f"{split}/test.npz"))

        print("res_raw = ", res_raw["acc_z"], res_raw["log_loss_z"])
        print("res_ft = ",  res_raw["acc_z"], res_ft["log_loss_z"])
        print("\n\n")
        
        res_list_raw.append(res_raw)
        res_list_ft.append(res_ft)

    res_pkl = {"raw":res_list_raw, "ft":res_list_ft}
    pickle.dump(res_pkl, open(res_path, "wb"))