from src import *
import matplotlib
matplotlib.use('Agg')


args = ModelOptions().parse()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

pw_network = PatchWiseNetwork(args.channels,init=False)
iw_network = ImageWiseNetwork(args.channels,init=False)

if args.network == '1':
    pw_model = PatchWiseModel(args, pw_network)
    pw_model.test(args.testset_path, verbose=True)

else:
    im_model = ImageWiseModel(args, iw_network, pw_network)
    im_model.test(args.testset_path, ensemble=args.ensemble == 1)
