import torch
import copy
import pandas as pd


def train(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss = 0

    for batch in data_loader:
      batch = batch.to(device)
      optimizer.zero_grad()
      out = model(batch)
      # ignore NaN targets (unlabeled) when computing training loss.
      is_labeled = batch.y == batch.y
      loss = loss_fn(out[is_labeled], batch.y[is_labeled])
      loss.backward()
      optimizer.step()

    return loss.item()


def eval(model, device, loader, evaluator, save_model_results=False):
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    if save_model_results:
        print ("Saving Model Predictions")
        
        # Create a pandas dataframe with a two columns
        # y_pred | y_true
        data = {}
        data['y_pred'] = y_pred.reshape(-1)
        data['y_true'] = y_true.reshape(-1)

        df = pd.DataFrame(data=data)
        df.to_csv('graph_evaluate.csv')

    return evaluator.eval(input_dict)



from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

if 'IS_GRADESCOPE_ENV' not in os.environ:
  model = GCN_Graph(4, args['hidden_dim'],
              1, args['num_layers'],
              args['dropout']).to(device)
  evaluator = Evaluator(name='ogbg-molhiv')

  dataset = PygGraphPropPredDataset(name='ogbg-molhiv')

  

if 'IS_GRADESCOPE_ENV' not in os.environ:
  model.reset_parameters()

  optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
  loss_fn = torch.nn.BCEWithLogitsLoss()

  best_model = None
  best_valid_acc = 0

  for epoch in range(1, 1 + args["epochs"]):
    print('Training...')
    loss = train(model, device, train_loader, optimizer, loss_fn)

    print('Evaluating...')
    train_result = eval(model, device, train_loader, evaluator)
    val_result = eval(model, device, valid_loader, evaluator)
    test_result = eval(model, device, test_loader, evaluator)

    train_acc, valid_acc, test_acc = train_result[dataset.eval_metric], val_result[dataset.eval_metric], test_result[dataset.eval_metric]
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_model = copy.deepcopy(model)
    print(f'Epoch: {epoch:02d}, '
          f'Loss: {loss:.4f}, '
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * valid_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')