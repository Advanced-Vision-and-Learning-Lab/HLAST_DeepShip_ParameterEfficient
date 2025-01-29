
def Parameters(args):

    histograms_shared = args.histograms_shared
    adapters_shared = args.adapters_shared
    lora_shared = args.lora_shared
    
    data_selection = args.data_selection
    Dataset_names = {0: 'DeepShip', 1: 'ShipsEar', 2: 'VTUAD'}
    
    #Number of bins for histogram layer. Recommended values are 4, 8 and 16.
    numBins = args.numBins
    RR = args.RR

    train_mode = args.train_mode
    use_pretrained = args.use_pretrained

    lr = args.lr

    parallel = True
    normalize_count = True
    normalize_bins = True

    batch_size = {'train': args.train_batch_size,'val': args.val_batch_size,'test': args.test_batch_size} 
    num_epochs = args.num_epochs
    
    patience = args.patience
    window_length = args.window_length
    hop_length = args.hop_length
    number_mels = args.number_mels
    
    pin_memory = False

    num_workers = args.num_workers

    feature = args.audio_feature
    
    Parallelize_model = True

    segment_length = args.segment_length

    sample_rate = args.sample_rate

    Model_name = args.model
    
    new_dir_p = './Datasets/DeepShip/'
    new_dir = '{}Segments_{}s_{}hz/'.format(new_dir_p,segment_length,sample_rate)
    
    #Return dictionary of parameters
    Params = {'histograms_shared': histograms_shared,'adapters_shared': adapters_shared,
                          'sample_rate':sample_rate,'segment_length':segment_length,'new_dir':new_dir,
                          'num_workers': num_workers,'lr': lr,'batch_size' : batch_size, 
                          'num_epochs': num_epochs,'normalize_count': normalize_count, 'data_selection':data_selection,
                          'normalize_bins': normalize_bins,'parallel': parallel,
                          'numBins': numBins,'RR': RR,'Model_name': Model_name, 
                          'train_mode': train_mode, 'use_pretrained': use_pretrained,
                          'pin_memory': pin_memory,'Parallelize': Parallelize_model,
                          'feature': feature, 'patience': patience,
                          'window_length':window_length,'hop_length':hop_length,'number_mels':number_mels,
                          'adapter_location': args.adapter_location,'adapter_mode': args.adapter_mode,
                          'histogram_location': args.histogram_location,'histogram_mode': args.histogram_mode,
                          'lora_target': args.lora_target, 'lora_rank': args.lora_rank, 'lora_shared': lora_shared}
    return Params

