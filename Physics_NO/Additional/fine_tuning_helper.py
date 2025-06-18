def freezing_parameters(model,model_architecture_,encoder_or_decoder='Encoder',freezing=0.5):
    '''Freezes the Parameters in the encoder or decoder.
       Warning: This function depends on the current CNO network'''

    downsampling_blocks=(model_architecture_["N_layers"]//2)*freezing
    res_batch_norm_in=(downsampling_blocks+1)*freezing
    res_batch_norm_inv=model_architecture_["N_layers"]*freezing
    Res_net_blocks=model_architecture_["N_res"]

    for name, param in model.named_parameters():
        parts = name.split(".")
        if parts[0]=='lift':
           param.requires_grad = False
        elif parts[0]=='batch_norm':
            if int(parts[1])<=res_batch_norm_in:
               param.requires_grad = False

        elif parts[0] == 'cont_conv_layers_invariant':
            if int(parts[1])<=downsampling_blocks:
                 param.requires_grad = False

        elif parts[0]=='res_batch_norm_in':
            param.requires_grad = False

        elif parts[0]=='resnet_blocks':
            if int(parts[1])%(Res_net_blocks)<= Res_net_blocks*freezing:
                  param.requires_grad = False
        
        elif parts[0]=='batch_norm_inv':
            if int(parts[1])<=res_batch_norm_inv:
                    param.requires_grad = False
        
        elif parts[0]=='cont_conv_layers':
            if int(parts[1])<=downsampling_blocks:
                    param.requires_grad = False


    if encoder_or_decoder=='Encoder':
        print(f'The encoder is frozen')

    elif encoder_or_decoder=='Decoder':
        for name, param in model.named_parameters():
                if param.requires_grad == False:
                     param.requires_grad = True
                else:
                     param.requires_grad = False
        print(f'The decoder is frozen')
    else:
         raise ValueError(f'encoder_or_decoder can not be {encoder_or_decoder}. Must be Encoder or Decoder')


             
