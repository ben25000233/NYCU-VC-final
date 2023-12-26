import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import cv2
from BARCNN import BARCNN
import math
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def patches_generator(image):
    [h, w] = image.shape
    patches = []
    patch_size = 32
    stride = 16
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            x = image[i:i + patch_size, j:j + patch_size]
            patches.append(x)

    return patches


def data_generator():
    current_path = os.getcwd()
    data_dir = current_path + "/train_rec"
    label_dir = current_path + "/train_ori"

    data, label = [], []
    for i in range(0, 3900):
        file_name = os.path.join(data_dir, "{}_rec.jpg".format(i))
        input_image = cv2.imread(file_name, 0)
        patches = patches_generator(input_image)

        label_file_name = os.path.join(label_dir, "{}.jpg".format(i))
        label_image = cv2.imread(label_file_name, 0)
        label_patches = patches_generator(label_image)

        data.append(patches)
        label.append(label_patches)


    data = np.array(data, dtype='uint8')
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3], 1))
    label = np.array(label, dtype='uint8')
    label = label.reshape((label.shape[0] * label.shape[1], label.shape[2], label.shape[3], 1))
  

    np.save(current_path + "/recovered_image_crop.npy", data)
    np.save(current_path + "/original_image_crop.npy", label)


    print("Data generation finished.")



def train(model, train_loader, val_loader, criterion, optimizer, epochs=100, device=device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            print(criterion(inputs, targets))
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            print(loss)
            exit()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_targets).item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {loss.item()}, Validation Loss: {val_loss}")
        
        current_path = os.getcwd()
        checkpoint_folder = os.path.join(current_path, "checkpoint")
        os.makedirs(checkpoint_folder, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_folder, f'BARCNN_epoch{epoch + 1}.pth')
        if(epoch +1)%10 == 0 :
            torch.save(model.state_dict(), checkpoint_path)


def test(model, device=device):
    current_path = os.getcwd()

    avg_pre_work_psnr = 0.0
    avg_before_model_psnr = 0.0
    avg_after_model_psnr = 0.0


    #load model
    checkpoint_path = os.path.join(current_path, "checkpoint", "BARCNN_epoch10.pth")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    for i in tqdm(range(100)):
        ori_image_path = os.path.join(current_path, "test_ori", f'{i+3900}.jpg')
        ori_image = cv2.imread(ori_image_path, 0)
        ori_image = np.float32(ori_image)
        '''
         pre_work_image_path = os.path.join(current_path, "pre_work_rec", f'{i+2000}_pre_work.jpg')
        pre_work_image = cv2.imread(pre_work_image_path, 0)
        pre_work_image = np.float32(pre_work_image)
        avg_pre_work_psnr += calculate_PSNR(pre_work_image, ori_image)
        '''
       


        rec_image_path = os.path.join(current_path, "test_rec", f'{i+3900}_rec.jpg')
        rec_image = cv2.imread(rec_image_path, 0)
        rec_image = np.float32(rec_image)
        avg_before_model_psnr += calculate_PSNR(rec_image, ori_image)
        print(check(rec_image, ori_image))
        
        # Convert the image to PyTorch tensor and format
        rec_image = torch.from_numpy(rec_image).unsqueeze(0).unsqueeze(0).float() / 255.0
  
        # Perform prediction
        with torch.no_grad():
            gpu_rec_image = rec_image.to(device)
            output = model(gpu_rec_image)
        # Post-process the output as needed (e.g., convert it to a NumPy array)
        pre_image = output.squeeze().cpu().numpy() * 255.0
        avg_after_model_psnr += calculate_PSNR(pre_image, ori_image)
        print(check(pre_image, ori_image))
        '''
         print("avg_before_model_psnr:", avg_before_model_psnr)
        print("avg_after_model_psnr:", avg_after_model_psnr)
        print(calculate_PSNR(pre_image, rec_image))
        '''
       
        print(check(pre_image, rec_image))
        # Save the predicted image
        output_path = os.path.join(current_path, "predict_image", f'{i+3900}_pre.jpg')
        cv2.imwrite(output_path, pre_image.astype('uint8'))
        exit()

    #print("avg_pre_work_psnr: ", avg_pre_work_psnr/100)  
    print("avg_before_model_psnr: ", avg_before_model_psnr/100)
    print("avg_after_model_psnr: ", avg_after_model_psnr/100)

def calculate_PSNR(reconstructed, original):
    
    original = np.array(original, dtype=np.uint32)
    pre_image = np.array(reconstructed, dtype=np.uint32)
    max_pixel_value = 255
    mse = np.mean((original - pre_image) ** 2)
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))
    return psnr

def check(img1, img2):
    img2 = np.float32(img2)
    img2 = img2.reshape(256, 256)
    print(img1.shape, img2.shape)
    num = 0
    for i in range(256):
        for j in range(256):
            if img1[i][j] != img2[i][j]:
                num+=1
    return num

def main():
    #train
    '''
    current_path = os.getcwd()
    data_generator()

    input_data = np.load(current_path + "/recovered_image_crop.npy").astype('float32') / 255.0
    label_data = np.load(current_path + "/original_image_crop.npy").astype('float32') / 255.0

    input_data = torch.from_numpy(input_data).permute(0, 3, 1, 2)  # Assuming input_data shape is [batch_size, 32, 32, 1]
    label_data = torch.from_numpy(label_data).permute(0, 3, 1, 2)  # Assuming label_data shape is [batch_size, 32, 32, 1]

    
    # Split the data into train and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(input_data, label_data, test_size=0.1, shuffle=True)


    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    model = BARCNN(nb_filters=1, depth=10)
    criterion = F.mse_loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train(model, train_loader, val_loader, criterion, optimizer, epochs=100)
    '''
    #test
    model = BARCNN(nb_filters=1, depth=10)
    test(model)
    
    

if __name__ == '__main__':
    main()
