import torch.autograd as autograd
import torch
import utils as utils


class LossReserveFixedLazySecondPrice():
    def __init__(self,reserve,distrib,nb_opponents=1):
        self.name = "LossReserveFixedLazySecondPrice"
        self.reserve = reserve
        self.nb_opponents = nb_opponents
        self.distrib = distrib
    def eval(self,net,input,size_batch):
          output = net(input)
          output_grad = autograd.grad(torch.sum(output),input,retain_graph=True,\
                    create_graph=True)[0]
          virtual_value_eval = utils.virtual_value(input,output,output_grad,self.distrib)
          indicator = torch.sigmoid(100*(output - self.reserve))
          winning = self.distrib.cdf(output)**self.nb_opponents
          loss = -1/size_batch*torch.sum(torch.mul(torch.mul(input - \
                        virtual_value_eval,winning),indicator))
          return loss
    def payment(self,net,input,size_batch):
          output = net(input)
          output_grad = autograd.grad(torch.sum(output),input,retain_graph=True,\
                    create_graph=True)[0]
          virtual_value_eval = utils.virtual_value(input,output,output_grad,self.distrib)
          indicator = torch.sigmoid(100*(output - self.reserve))
          winning = self.distrib.cdf(output)**self.nb_opponents
          loss = -1/size_batch*torch.sum(torch.mul(torch.mul(\
                        virtual_value_eval,winning),indicator))
          return loss


class LossReserveFixedEagerSecondPrice():
    def __init__(self,reserve,distrib,nb_opponents=1):
        self.name = "LossReserveFixedEagerSecondPrice"
        self.reserve = reserve
        self.nb_opponents = nb_opponents
        self.distrib = distrib
    def eval(self,net,input,size_batch):
          output = net(input)
          output_grad = autograd.grad(torch.sum(output),input,retain_graph=True,\
                    create_graph=True)[0]
          virtual_value_eval = utils.virtual_value(input,output,output_grad,self.distrib)
          print(virtual_value_eval)
          print(self.reserve)
          print(output)
          indicator = torch.sigmoid(100000*(output - self.reserve))
          print(indicator)
          winning = torch.max(self.distrib.cdf(torch.tensor(self.distrib.optimal_reserve_price)),\
                self.distrib.cdf(output))**self.nb_opponents

          loss = -1/size_batch*torch.sum(torch.mul(torch.mul(input - \
                        virtual_value_eval,winning),indicator))
          return loss
    def payment(self,net,input,size_batch):
          output = net(input)
          output_grad = autograd.grad(torch.sum(output),input,retain_graph=True,\
                    create_graph=True)[0]
          virtual_value_eval = utils.virtual_value(input,output,output_grad,self.distrib)
          indicator = torch.sigmoid(100*(output - self.reserve))
          winning = torch.max(self.distrib.cdf(torch.tensor(self.distrib.optimal_reserve_price)),\
                self.distrib.cdf(output))**self.nb_opponents

          loss = -1/size_batch*torch.sum(torch.mul(torch.mul(\
                        virtual_value_eval,winning),indicator))
          return loss


class LossMonopolyReserveLazySecondPrice():
    def __init__(self,distrib,nb_opponents=1):
        self.name = "LossMonopolyReserveLazySecondPrice"
        self.nb_opponents = nb_opponents
        self.distrib = distrib
    def eval(self,net,input,size_batch):
          output = net(input)
          output_grad = autograd.grad(torch.sum(output),input,retain_graph=True,\
                        create_graph=True)[0]
          virtual_value_eval = utils.virtual_value(input,output,output_grad,self.distrib)
          index_reserve_price = torch.where(virtual_value_eval < 0.0)[0]
          if len(index_reserve_price) == 0:
              reserve_price = net(torch.Tensor([0.0]))
              reserve_value = 0.0
          else:
              reserve_value = torch.max(torch.index_select(input, 0, index_reserve_price))
              reserve_price = torch.max(torch.index_select(output, 0, index_reserve_price))
          indicator = torch.sigmoid(1000*(input - reserve_value))*torch.sigmoid(1000*(virtual_value_eval-0.001))*\
                      torch.sigmoid(1000*(output - reserve_price))
          winning = self.distrib.cdf(output)**self.nb_opponents
          loss = -1/size_batch*torch.sum(torch.mul(torch.mul(input - \
           virtual_value_eval,torch.min(winning,torch.tensor(1.0))),indicator))
          return loss
    def payment(self,net,input,size_batch):
          output = net(input)
          output_grad = autograd.grad(torch.sum(output),input,retain_graph=True,\
                        create_graph=True)[0]
          virtual_value_eval = utils.virtual_value(input,output,output_grad,self.distrib)
          index_reserve_price = torch.where(virtual_value_eval < 0.0)[0]
          if len(index_reserve_price) == 0:
              reserve_price = net(torch.Tensor([0.0]))
              reserve_value = 0.0
          else:
              reserve_value = torch.max(torch.index_select(input, 0, index_reserve_price))
              reserve_price = torch.max(torch.index_select(output, 0, index_reserve_price))
          indicator = torch.sigmoid(1000*virtual_value_eval)*torch.sigmoid(1000*(input - reserve_value))*\
                      torch.sigmoid(1000*(output - reserve_price))
          winning = self.distrib.cdf(output)**self.nb_opponents
          loss = -1/size_batch*torch.sum(torch.mul(torch.mul(\
           virtual_value_eval,torch.min(winning,torch.tensor(1.0))),indicator))
          return loss



class LossMonopolyReserveEagerSecondPrice():
    def __init__(self,distrib,nb_opponents=1):
        self.name = "LossMonopolyReserveEagerSecondPrice"
        self.nb_opponents = nb_opponents
        self.distrib = distrib
    def eval(self,net,input,size_batch):
          output = net(input)
          output_grad = autograd.grad(torch.sum(output),input,retain_graph=True,\
                        create_graph=True)[0]
          virtual_value_eval = utils.virtual_value(input,output,output_grad,self.distrib)
          index_reserve_price = torch.where(virtual_value_eval<0.0)[0]
          if len(index_reserve_price) == 0:
              reserve_price = net(torch.Tensor([0.0]))
              reserve_value = 0.0
          else:
            reserve_value = torch.max(torch.index_select(input,0,index_reserve_price))
            reserve_price = torch.max(torch.index_select(output,0,index_reserve_price))
          indicator = torch.sigmoid(1000*(input - reserve_value))*torch.sigmoid(1000*virtual_value_eval)*\
                      torch.sigmoid(1000*(output - reserve_price))
          winning = torch.max(self.distrib.cdf(torch.tensor(self.distrib.optimal_reserve_price)),\
          self.distrib.cdf(output))**self.nb_opponents
          loss = -1/size_batch*torch.sum(torch.mul(torch.mul(input - \
            virtual_value_eval,torch.min(winning,torch.tensor(1.0))),indicator))
          return loss
    def payment(self,net,input,size_batch):
          output = net(input)
          output_grad = autograd.grad(torch.sum(output),input,retain_graph=True,\
                        create_graph=True)[0]
          virtual_value_eval = utils.virtual_value(input,output,output_grad,self.distrib)
          indicator = torch.sigmoid(1000*(virtual_value_eval-0.001))
          winning = torch.max(self.distrib.cdf(torch.tensor(self.distrib.optimal_reserve_price)),\
          self.distrib.cdf(output))**self.nb_opponents
          loss = -1/size_batch*torch.sum(torch.mul(torch.mul(\
           virtual_value_eval,torch.min(winning,torch.tensor(1.0))),indicator))
          return loss

class lossMyersonAuction():
    def __init__(self,distrib,nb_opponents=1):
        self.name = "LossPersonalizedReserve"
        self.nb_opponents = nb_opponents
        self.distrib =distrib

    def eval(self,net,input,size_batch,nb_opponents=3):
          output = net(input)
          output_grad = autograd.grad(torch.sum(output),input,retain_graph=True,\
                    create_graph=True)[0]

          virtual_value_eval = utils.virtual_value(input, output, output_grad, self.distrib)
          index_reserve_price = torch.where(virtual_value_eval < 0.0)[0]
          if len(index_reserve_price) == 0:
              reserve_price = net(torch.Tensor([0.0]))
              reserve_value = 0.0
          else:
              reserve_value = torch.max(torch.index_select(input, 0, index_reserve_price))
              reserve_price = torch.max(torch.index_select(output, 0, index_reserve_price))
          indicator = torch.sigmoid(1000 * (input - reserve_value)) * torch.sigmoid(
              1000 * virtual_value_eval) * torch.sigmoid(1000 * (output - reserve_price))
          winning = torch.max(self.distrib.cdf(torch.tensor(self.distrib.optimal_reserve_price)),\
          self.distrib.cdf(self.distrib.inverse_virtual_value(virtual_value_eval)))**self.nb_opponents
          loss = -1/size_batch*torch.sum(torch.mul(torch.mul(input - \
                virtual_value_eval,torch.min(winning,\
                    torch.tensor(1.0))),indicator))
          return loss


    def payment(self,net,input,size_batch,nb_opponents=3):
          output = net(input)
          output_grad = autograd.grad(torch.sum(output),input,retain_graph=True,\
                    create_graph=True)[0]
          virtual_value_eval = utils.virtual_value(input, output, output_grad, self.distrib)
          index_reserve_price = torch.where(virtual_value_eval < 0.0)[0]
          if len(index_reserve_price) == 0:
              reserve_price = net(torch.Tensor([0.0]))
              reserve_value = 0.0
          else:
              reserve_value = torch.max(torch.index_select(input, 0, index_reserve_price))
              reserve_price = torch.max(torch.index_select(output, 0, index_reserve_price))
          indicator = torch.sigmoid(1000 * (input - reserve_value)) * torch.sigmoid(
              1000 * virtual_value_eval) * torch.sigmoid(1000 * (output - reserve_price))
          winning = torch.max(self.distrib.cdf(torch.tensor(self.distrib.optimal_reserve_price)),\
          self.distrib.cdf(self.distrib.inverse_virtual_value(virtual_value_eval)))**self.nb_opponents
          loss = -1/size_batch*torch.sum(torch.mul(torch.mul(\
                virtual_value_eval,torch.min(winning,\
                    torch.tensor(1.0))),indicator))
          return loss


class lossBoostedSecondPriceLinearFit():
    def __init__(self,distrib,nb_opponents=1):
        self.name = "BoostedSecondPriceLinear"
        self.nb_opponents = nb_opponents
        self.distrib = distrib
    def eval(self,net,input,size_batch):
          output = net(input)

          output_grad = autograd.grad(torch.sum(output),input,\
                    retain_graph=True,create_graph=True)[0]

          virtual_value_eval = utils.virtual_value(input,output,output_grad,self.distrib)
          index_reserve_price = torch.where(virtual_value_eval < 0.0)[0]
          if len(index_reserve_price) == 0:
              reserve_price = net(torch.Tensor([0.0]))
              reserve_value = 0.0
          else:
              reserve_value = torch.max(torch.index_select(input, 0, index_reserve_price))
              reserve_price = torch.max(torch.index_select(output, 0, index_reserve_price))

          indicator = torch.sigmoid(1000 * (input - reserve_value)) * torch.sigmoid(1000 * (virtual_value_eval - 0.001)) * \
                      torch.sigmoid(1000 * (output - reserve_price))
          # fit a corresponding to the linear fit above the reserve price
          input_index = torch.where(input>=reserve_value)[0]
          input_fit =  torch.index_select(input, 0, input_index)
          virtual_value_eval_restrict = torch.index_select(virtual_value_eval, 0, input_index)

          mean_input = torch.mean(input_fit)
          mean_virtual_value = torch.mean(virtual_value_eval_restrict)

          a = torch.sum(torch.mul(virtual_value_eval_restrict - mean_virtual_value, input_fit - mean_input)) \
              / torch.sum(torch.mul(input_fit - mean_input, input_fit - mean_input))
          b = mean_virtual_value - a * mean_input
          virtual_value_fit = a * input + b
          #compute probability of winning
          winning = torch.max(self.distrib.cdf(torch.tensor(self.distrib.optimal_reserve_price)),self.distrib.cdf(virtual_value_fit/self.distrib.boost))\
                    **self.nb_opponents
          loss = -1/size_batch*torch.sum(torch.mul(torch.mul(input - torch.max(virtual_value_fit,torch.tensor(0.0)),\
          torch.min(winning,\
                    torch.tensor(1.0))),indicator))
          return loss

    def payment(self,net,input,size_batch):
          output = net(input)

          output_grad = autograd.grad(torch.sum(output),input,\
                    retain_graph=True,create_graph=True)[0]

          virtual_value_eval = utils.virtual_value(input,output,output_grad,self.distrib)
          index_reserve_price = torch.where(virtual_value_eval < 0.0)[0]
          if len(index_reserve_price) == 0:
              reserve_price = net(torch.Tensor([0.0]))
              reserve_value = 0.0
          else:
              reserve_value = torch.max(torch.index_select(input, 0, index_reserve_price))
              reserve_price = torch.max(torch.index_select(output, 0, index_reserve_price))

          indicator = torch.sigmoid(1000 * (input - reserve_value)) * torch.sigmoid(
              1000 * (virtual_value_eval)) * \
                      torch.sigmoid(1000 * (output - reserve_price))

          # fit a corresponding to the linear fit above the reserve price
          input_index = torch.where(input >= reserve_value)[0]
          input_fit = torch.index_select(input, 0, input_index)

          virtual_value_eval_restrict = torch.index_select(virtual_value_eval, 0, input_index)

          mean_input = torch.mean(input_fit)
          mean_virtual_value = torch.mean(virtual_value_eval_restrict)

          a = torch.sum(torch.mul(virtual_value_eval_restrict - mean_virtual_value, input_fit - mean_input)) \
              / torch.sum(torch.mul(input_fit - mean_input, input_fit - mean_input))
          b = mean_virtual_value - a * mean_input
          virtual_value_fit = a * input + b
          #compute probability of winning
          winning = torch.max(self.distrib.cdf(torch.tensor(self.distrib.optimal_reserve_price)),self.distrib.cdf(virtual_value_fit/self.distrib.boost))\
                    **self.nb_opponents
          loss = -1/size_batch*torch.sum(torch.mul(torch.mul(torch.max(virtual_value_fit,torch.tensor(0.0)),\
          torch.min(winning,\
                    torch.tensor(1.0))),indicator))
          return loss

class lossBoostedSecondPriceAffineFit():
    def __init__(self,distrib,nb_opponents=1):
        self.name = "BoostedSecondPriceAffine"
        self.nb_opponents = nb_opponents
        self.distrib = distrib
    def eval(self,net,input,size_batch):
          output = net(input)

          output_grad = autograd.grad(torch.sum(output),input,\
                    retain_graph=True,create_graph=True)[0]

          virtual_value_eval = utils.virtual_value(input,output,output_grad,self.distrib)

          #fit a,b corresponding to boosted second price
          mean_psi = torch.mean(virtual_value_eval)
          mean_bid = torch.mean(output)
          a = torch.sum(torch.mul(virtual_value_eval - mean_psi,output - mean_bid)) \
            /torch.sum(torch.mul(output - mean_bid,output - mean_bid))
          b = mean_psi - a * mean_bid
          virtual_value_fit = a*output + b

          #use affine fit to define the reserve price
          indicator = torch.sigmoid(1000*(virtual_value_fit))

          #compute the probability of winning
          winning = torch.max(self.distrib.cdf(torch.tensor(self.distrib.optimal_reserve_price)),\
          self.distrib.cdf(self.distrib.inverse_virtual_value(virtual_value_fit)))**self.nb_opponents

          loss = -1/size_batch*torch.sum(torch.mul(torch.mul(input - virtual_value_fit,\
          torch.min(winning,torch.tensor(1.0))),indicator))
          return loss
    def payment(self,net,input,size_batch):
          output = net(input)

          output_grad = autograd.grad(torch.sum(output),input,\
                    retain_graph=True,create_graph=True)[0]

          virtual_value_eval = utils.virtual_value(input,output,output_grad,self.distrib)

          #fit a,b corresponding to boosted second price
          mean_psi = torch.mean(virtual_value_eval)
          mean_bid = torch.mean(output)
          a = torch.sum(torch.mul(virtual_value_eval - mean_psi,output - mean_bid)) \
            /torch.sum(torch.mul(output - mean_bid,output - mean_bid))
          b = mean_psi - a * mean_bid
          virtual_value_fit = a*output + b

          #use affine fit to define the reserve price
          indicator = torch.sigmoid(1000*(virtual_value_fit))

          #compute the probability of winning
          winning = torch.max(self.distrib.cdf(torch.tensor(self.distrib.optimal_reserve_price)),\
          self.distrib.cdf(self.distrib.inverse_virtual_value(virtual_value_fit)))**self.nb_opponents

          loss = -1/size_batch*torch.sum(torch.mul(torch.mul(virtual_value_fit,\
          torch.min(winning,torch.tensor(1.0))),indicator))
          return loss