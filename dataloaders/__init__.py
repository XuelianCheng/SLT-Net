from torch.utils.data import DataLoader
from dataloaders.video_list import get_loader, test_dataset
from dataloaders.video_list_long import get_loader as get_loader_long
from dataloaders.video_list_long import test_dataset as test_dataset_long

# dataloader for video COD
def video_dataloader(args):          
    train_loader = get_loader(dataset=args.dataset,
                              batchsize=args.batchsize,
                              trainsize=args.trainsize,
                              train_split=args.trainsplit,
                              num_workers=8)
    val_loader = test_dataset(dataset=args.dataset,
                              testsize=args.trainsize)
    print('Training with %d image pairs' % len(train_loader))
    print('Val with %d image pairs' % len(val_loader))
    return train_loader, val_loader 

def video_dataloader_long(args):          
    train_loader = get_loader_long(dataset=args.dataset,
                              batchsize=args.batchsize,
                              trainsize=args.trainsize,
                              input_length=args.input_length,
                              fsampling_rate=args.fsampling_rate,
                              num_workers=args.threads)
    val_loader = test_dataset_long(dataset=args.dataset,
                              input_length=args.input_length,
                              fsampling_rate=args.fsampling_rate)
    print('Training with %d image pairs' % len(train_loader))
    print('Val with %d image pairs' % len(val_loader))
    return train_loader, val_loader 

def test_dataloader(args):   
    test_loader = test_dataset(dataset=args.dataset,
                              split=args.testsplit,
                              testsize=args.testsize)
    print('Test with %d image pairs' % len(test_loader))
    return test_loader 