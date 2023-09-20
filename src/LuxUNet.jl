module LuxUNet

using Reexport
using Lux
@reexport using Lux



function ConvBlock(in_c, out_c; activation=relu, kernel_size=(3,3))
    layers = [Conv(kernel_size, in_c => out_c, identity, pad=(1,1)),
              BatchNorm(out_c),
              Conv(kernel_size, out_c => out_c, identity, pad=(1,1)),
              BatchNorm(out_c, relu)]

    return Chain(layers...)
end


function EncoderBlock(in_c, out_c; activation=relu, kernel_size=(3,3), pool_size=(2,2), pool_layer=MaxPool)
    cb = ConvBlock(in_c, out_c; activation, kernel_size)
    pl = pool_layer(pool_size)
    return Chain(cb, pl)
end


end
