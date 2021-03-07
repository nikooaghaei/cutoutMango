from util.MangoBox import load_from, run_mango
from util.model_tools import train_VGG#, train_simple
from util.data import set_data_CIFAR10

batch_size = 32
num_of_epochs = 100

#### LOADING DATA ####
trainloader, testloader = set_data_CIFAR10(batch_size)

#### TRAINING MODEL ####
model = train_VGG(trainloader, testloader, batch_size, num_of_epochs)
                #   load_path='models/vggcnn.pt')
# model = train_and_test(trainloader, stestloader, "vanilla_model.pt",
#                        num_of_epochs, save=True)                  

#### CREATING MANGO DATA ####
mango_trainloader = run_mango(model, trainloader, 
                            #   load_from="data/MANGO/t_train/maskD.txt",
                              folder_name='t_train',
                              batch_size=batch_size,
                              n_workers=2)

#### TRAINING WITH MANGO DATA ####
model = train_VGG(mango_trainloader, testloader, batch_size, num_of_epochs)
# model = train_and_test(new_trainloader, testloader, "mango_model.pt",
#                        num_of_epochs, True)