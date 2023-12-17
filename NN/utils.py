import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import torch.autograd as autograd
import numpy as np
import bidder_strategy as learning
import losses


def virtual_value(value, bid, bid_grad, distrib):
    return bid - bid_grad * (1 - distrib.cdf(value)) / distrib.pdf(value)


def plot_strategy(strategy, distrib, nb_points=1000):
    x = torch.zeros((nb_points), requires_grad=True)
    samples = distrib.sample((nb_points, 1))
    x.data = samples.clone()
    output = strategy(x)
    value = x.reshape((nb_points)).detach().numpy()
    bid = output.reshape((nb_points)).detach().numpy()
    bid = bid[value.argsort()]
    value = np.sort(value)

    plt.plot(value, bid, lw=3)
    plt.title("Bidding strategy")
    plt.xlabel('Value')
    plt.ylabel('Bid')
    plt.savefig('strategy.pdf')
    plt.show()


def plot_virtual_value(net, distrib, loss_eval=0, nb_points=1000):
    x = torch.zeros((nb_points), requires_grad=True)
    samples = distrib.sample((nb_points, 1))
    x.data = samples.clone()
    output = net(x)
    output_grad = autograd.grad(torch.sum(output), x, retain_graph=True)[0]
    virtual_value_eval = virtual_value(x, output, output_grad, distrib)
    value = x.reshape((nb_points)).detach().numpy()
    virtual_value_eval = virtual_value_eval.reshape((nb_points)).detach().numpy()
    virtual_value_eval = virtual_value_eval[value.argsort()]
    value = np.sort(value)

    plt.plot(value, virtual_value_eval, lw=3)
    plt.xlabel('Value')
    plt.ylabel('Virtual value')
    plt.title('Virtual value')
    plt.savefig('virtual_value.pdf')
    plt.show()


def plot_loss(nb_steps, loss_list):
    plt.plot(np.arange(nb_steps), loss_list)
    plt.title("Evolution of the loss")
    plt.xlabel("Number of steps")
    plt.ylabel("Loss")
    plt.savefig('loss.pdf')
    plt.show()


def compute_affine_regression(input, net, distrib):
    output = net(input)

    output_grad = autograd.grad(torch.sum(output), input, retain_graph=True, create_graph=True)[0]

    virtual_value_eval = virtual_value(input, output, output_grad, distrib)
    mean_psi = torch.mean(virtual_value_eval)
    mean_bid = torch.mean(output)
    a = torch.sum(torch.mul(virtual_value_eval - mean_psi, output - mean_bid)) \
        / torch.sum(torch.mul(output_grad - mean_bid, output_grad - mean_bid))
    b = mean_psi - a * mean_bid
    print(f"a:{a}")
    print(f"b:{b}")
    return a, b


def compute_linear_regression(input, net, distrib):
    output = net(input)

    output_grad = autograd.grad(torch.sum(output), input, retain_graph=True, create_graph=True)[0]

    virtual_value_eval = virtual_value(input, output, output_grad, distrib)

    indicator = torch.sigmoid(1000 * (virtual_value_eval))
    mean_psi = torch.sum(virtual_value_eval * indicator) / torch.sum(indicator)
    mean_value = torch.sum(input * indicator) / torch.sum(indicator)
    min_fit = torch.min(torch.where((indicator > 0.5), input, torch.tensor(1.0)))
    a = mean_psi / (mean_value - min_fit)
    print(f"min_fit:{min_fit}")
    print(f"a:{a}")
    return a


def run_multiple_runs_second_price(nb_runs, size_batch_eval, loss_net_train, loss_net_eval, distrib,
                                   nb_opponents=1, nb_steps_learning=2000, size_batch_learning=20000,
                                   lr=0.0001,auction_type='lazy_second_price'):
    perf_summary = []
    training_perf_summary = []
    revenue_summary = []
    for i in range(nb_runs):
        net = learning.Net_one_layer()
        strategy = learning.main(net, loss_net_train, nb_steps=nb_steps_learning, size_batch=size_batch_learning, lr=lr)
        training_perf = sanity_check_second_price(strategy, loss_net_train, size_batch_eval)
        real_perf = sanity_check_second_price(strategy, loss_net_eval, size_batch_eval)
        revenue = compute_revenue(strategy, loss_net_train, size_batch_eval, distrib, auction_type, nb_opponents)
        training_perf_summary.append(training_perf)
        perf_summary.append(real_perf)
        revenue_summary.append(revenue)

    index_opt_train = np.argmax(training_perf_summary)
    best_utility = perf_summary[index_opt_train]
    revenue_when_best_utility = revenue_summary[index_opt_train]

    mean = np.mean(perf_summary)
    std = np.std(perf_summary)
    revenue_mean = np.mean(revenue_summary)
    revenue_std = np.std(revenue_summary)
    result = {"mean_utility": mean, "std_utility": std,"best_utility":best_utility,
              "mean_revenue": revenue_mean, "std_revenue": revenue_std,
              'revenue_when_best_utility':revenue_when_best_utility}
    return result


def sanity_check_second_price(strategy, loss_eval, nb_samples):
    x = torch.zeros((nb_samples), requires_grad=True)
    grid = torch.linspace(0, 1, steps=nb_samples)
    x.data = grid.clone()
    x = x.reshape((nb_samples, 1))
    output = strategy(x)
    output_grad = autograd.grad(torch.sum(output), x, retain_graph=True)[0]
    virtual_value_eval = virtual_value(x, output, output_grad, loss_eval.distrib)
    virtual_value_eval = virtual_value_eval.reshape((nb_samples)).detach().numpy()
    value = x.reshape((nb_samples)).detach().numpy()

    # we are conservative here by assuming that the reserve value is
    # the maximum value where the virtual value is negative
    if len(value[virtual_value_eval <= 0.0]) > 0:
        x_0 = np.max(value[virtual_value_eval <= 0.0])
        print(x_0)
        x_0 = torch.tensor(x_0).reshape((1, 1)).float()
        reserve_price = strategy(x_0)
    else:
        x_0 = 0.0
        x_0 = torch.tensor(x_0).reshape((1, 1)).float()
        reserve_price = torch.min(output)
    print(f"reserve_value:{x_0}")
    print(f"reserve price:{reserve_price})")
    # compute the real performance of the strategy
    loss_eval.reserve = reserve_price
    perf = compute_loss(strategy, loss_eval, nb_samples)
    print(f"perf:{perf}")
    return perf


def run_multiple_runs_simple(nb_runs, loss_net, size_batch_eval, nb_steps_train=2000, size_batch_train=20000,
                             lr=0.0001):
    perf_summary = []
    for i in range(nb_runs):
        net = learning.Net_one_layer()
        learning.main(net, loss_net, nb_steps=nb_steps_train, size_batch=size_batch_train, lr=lr)
        perf_summary.append(compute_loss(net, loss_net, size_batch_eval))
    mean = np.mean(perf_summary)
    std = np.std(perf_summary)
    result = {"mean": mean, "std": std}
    return result


def compute_loss(net, loss, nb_samples):
    input = torch.zeros((nb_samples, 1), requires_grad=True)
    samples = loss.distrib.sample((nb_samples, 1))
    input.data = samples.clone()
    loss_eval = loss.eval(net, input, nb_samples)
    loss_eval = -loss_eval.detach().numpy()
    return loss_eval


def ironing_psi_for_myerson():
    return True


def compute_reserve_price(strategy, loss_eval, nb_samples):
    x = torch.zeros((nb_samples), requires_grad=True)
    grid = torch.linspace(0, 1, steps=nb_samples)
    x.data = grid.clone()
    x = x.reshape((nb_samples, 1))
    output = strategy(x)
    output_grad = autograd.grad(torch.sum(output), x, retain_graph=True)[0]
    virtual_value_eval = virtual_value(x, output, output_grad, loss_eval.distrib)
    virtual_value_eval = virtual_value_eval.reshape((nb_samples)).detach().numpy()
    value = x.reshape((nb_samples)).detach().numpy()

    # we are conservative here by assuming that the reserve value is
    # the maximum value where the virtual value is negative
    if len(value[virtual_value_eval <= 0.0]) > 0:
        x_0 = np.max(value[virtual_value_eval <= 0.0])
        print(x_0)
        x_0 = torch.tensor(x_0).reshape((1, 1)).float()
        reserve_price = strategy(x_0)
    else:
        x_0 = 0.0
        x_0 = torch.tensor(x_0).reshape((1, 1)).float()
        reserve_price = strategy(x_0)
    return reserve_price


def run_auction(auction_type, list_bids, list_virtualized_bids, nb_bidders, reserve_prices, distrib):
    if auction_type == "lazy_second_price":
        winner = np.random.choice((np.flatnonzero(list_bids == max(list_bids))))
        if winner > 0:
            if list_bids[winner] >= reserve_prices[winner]:
                payment = max(np.sort(list_bids)[-2], reserve_prices[winner])
                return payment
            else:
                return 0.0
        else:
            return 0.0
    elif auction_type == "eager_second_price":
        for j in range(nb_bidders):
            if list_bids[j] < reserve_prices[j]:
                list_bids[j] = 0
        winner = np.random.choice((np.flatnonzero(list_bids == max(list_bids))))
        if winner > 0:
            if list_bids[winner] >= reserve_prices[winner]:
                payment = max(np.sort(list_bids)[-2], reserve_prices[winner])
                return payment
            else:
                return 0.0
        else:
            return 0.0
    elif auction_type == "Myerson" or auction_type == "boosted_second_price":
        winner = np.random.choice((np.flatnonzero(list_virtualized_bids == max(list_virtualized_bids))))
        if winner > 0:
            if list_virtualized_bids[winner] >= 0:
                payment = np.sort(list_virtualized_bids)[-2]
                if payment > 0:
                    payment = distrib.inverse_virtual_value(payment)
                else:
                    payment = reserve_prices[winner]
                return payment
            else:
                return 0.0
        else:
            return 0.0

    else:
        print("wrong type auction")


def compute_revenue(strategy, loss, nb_samples, distrib, auction_type, nb_opponents=1):
    nb_players = nb_opponents + 1
    x = torch.zeros(nb_samples, requires_grad=True)
    samples = distrib.sample((nb_samples, 1))
    x.data = samples.clone()
    output = strategy(x)
    output_grad = autograd.grad(torch.sum(output), x, retain_graph=True)[0]
    bid = output.reshape(nb_samples).detach().numpy()
    payment_strategic_bidder = loss.payment(strategy, x, nb_samples)
    payment_strategic_bidder = -payment_strategic_bidder.detach().numpy()

    virtual_value_eval = virtual_value(x, output, output_grad, distrib)
    virtual_value_eval = virtual_value_eval.reshape(nb_samples).detach().numpy()

    reserve_price = compute_reserve_price(strategy, loss, nb_samples)
    bid_table = np.zeros((nb_samples, nb_players))
    virtualized_bid_table = np.zeros((nb_samples, nb_players))

    bid_table[:, 0] = bid
    for i in range(nb_players):
        if i == 0:
            bid_table[:, 0] = bid
        else:
            bid_table[:, i] = np.random.uniform(size=[nb_samples])
        if auction_type == 'boosted_second_price':
            if i == 0:
                a, b = compute_affine_regression(x, strategy, distrib)
                a = a.detach().numpy()
                b = b.detach().numpy()
                virtualized_bid_table[:, i] = a * bid_table[:, i] + b
            else:
                virtualized_bid_table[:, i] = distrib.virtual_value(bid_table[:, i])
        else:
            if i == 0:
                virtualized_bid_table[:, i] = virtual_value_eval
            else:
                virtualized_bid_table[:, i] = distrib.virtual_value(bid_table[:, i])

    reserve_prices = 0.5 * np.ones(nb_players)
    reserve_prices[0] = reserve_price
    payment_other_bidders = 0

    for j in range(nb_samples):
        list_bids = bid_table[j, :]
        list_virtualized_bids = virtualized_bid_table[j, :]
        payment_other_bidders += run_auction(auction_type, list_bids, list_virtualized_bids, nb_players, reserve_prices, distrib)

    revenue = payment_other_bidders/nb_samples + payment_strategic_bidder
    print(f'payment :{payment_strategic_bidder}')
    print(f'revenue: {revenue}')
    return revenue
