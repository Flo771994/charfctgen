import torch


# Define the model class
class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, layers_data: list):
        """ creates a fully connected neural network with user-specified activation functions
            input_size is dimension of the input
            the hidden layer architecture is passed via the layers_data list, which must consist of tuples (size,activation)
            size is the output dimension of the hidden layer and activation is the activation function that is applied ot each neuron in the output of the hidden layer
            the output of the neural net is always of dimension output_size and is obtained by applying a linear layer (no activation function applied)
        """
        super().__init__() #call constructor of parent

        self.layers = torch.nn.ModuleList() #initialize module list
        self.input_size = input_size #store input_size
        self.output_size = output_size #store output_size
        self.kwargs = {'input_size': input_size, 'output_size': output_size, 'layers_data': layers_data}
        #create hidden layers
        for size, activation in layers_data: #iterate over the hidden layers_data that contain the size and activation function of each hidden layer
            self.layers.append(torch.nn.Linear(input_size, size))
            if activation is not None:
                assert isinstance(activation, torch.nn.modules.Module), \
                    "Each tuple should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation) #set activation function for the hidden layer
            input_size = size  # input_size of the next layer

        #create output layer
        self.layers.append(torch.nn.Linear(input_size,output_size)) #define the linear output layer
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.to(self.device)


    def forward(self, input_data):
        #iterate through layers ModuleList and apply the functions to the input_data
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data

