import gurobipy as gp
model_path="GUR_TEST_model_name.mps"
branch_file="GUR_TEST_branch_priorities_2.txt"

options = {
        "WLSACCESSID": "b7836a23-3df1-40ac-be4d-310282e2178e",
        "WLSSECRET": "8dd2c11c-cb9b-46f3-b072-4887712ea0c9",
        "LICENSEID": 2690165
    }


min_priority_keep_integer=1000
max_priority_keep_integer=1000

terms_keep=set([])
#terms_keep.add(int(1))
#terms_keep.add(int(50))
for i in range(min_priority_keep_integer,max_priority_keep_integer+1):
    terms_keep.add(int(i))

with gp.Env(params=options) as env:
    with gp.read(model_path, env=env) as model:
        with open(branch_file, "r") as f:
            for line in f:
                name, bp = line.strip().split()
                var = model.getVarByName(name)
                if int(bp) not in terms_keep:#<min_priority_keep_integer or int(bp)>max_priority_keep_integer:
                    if var.vType==gp.GRB.BINARY:
                        var.Ub=1
                    var.vType=gp.GRB.CONTINUOUS
                else:
                    var.BranchPriority=int(bp)
            model.update()
            model.optimize()