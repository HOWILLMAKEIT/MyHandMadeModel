import torch 
import torch.nn as nn
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(self,in_channels,patch_size,embed_dim,img_size):
        super().__init__()
        num_patches = (img_size//patch_size) ** 2
        self.patcher = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )

        self.cls_token = nn.Parameter(torch.randn(size=(1,1,embed_dim)),requires_grad=True)
        self.position_embedding =  nn.Parameter(torch.randn(size = (1,num_patches+1,embed_dim),requires_grad = True))
        
    def forward(self,x):
        cls_token = self.cls_token.expand(x.shape[0],-1,-1)

        x = self.patcher(x)
        x = torch.cat([cls_token,x],dim = 1)
        x = x + self.position_embedding
        return x
    


class Vit(nn.Module):
    def __init__(self,in_channels,patch_size,embed_dim,img_size,
                 num_heads,activation,dropout,num_encoderss,num_classes):
        super().__init__()  
        self.patch_embedding = PatchEmbedding(in_channels,patch_size,embed_dim,img_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model= embed_dim,nhead=num_heads,
                                                   activation=activation,dropout=dropout,batch_first=True,norm_first=True)
        self.encoder_layer = nn.TransformerEncoder(encoder_layer,num_encoderss)
        self.MLP = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim,out_features=num_classes)
        )
    def forward(self,x):
        x = self.patch_embedding(x)
        x = self.encoder_layer(x)
        x = self.MLP(x[:,0,:])
        return x 


# 单元测试
def test_vit_output_shape():
    """测试VIT模型输出维度是否正确"""
    test_x = torch.randn(32,3,224,224)
    model = Vit(3,16,756,224,12,nn.GELU(),0.05,12,10)
    device = torch.device('cuda')
    model = model.to(device)
    test_x = test_x.to(device)
    test_y = model(test_x)
    
    # 验证输出维度是否为[batch_size, num_classes]
    assert test_y.shape == (32, 10), f"输出维度错误,应为[32,10],实际为{test_y.shape}"
    print("VIT模型输出维度测试通过!")
def test_patch_embedding_output_shape():
    """测试PatchEmbedding模块输出维度是否正确"""
    test_x = torch.randn(256,3,224,224)
    model = PatchEmbedding(3,16,756,224)
    test_y = model(test_x)
    
    # 验证输出维度是否为[batch_size, num_patches+1, embed_dim]
    assert test_y.shape == (256,197,756), f"输出维度错误,应为[256,197,756],实际为{test_y.shape}"
    print("PatchEmbedding模块输出维度测试通过!")
if __name__ == "__main__":
    test_vit_output_shape()
    test_patch_embedding_output_shape()


