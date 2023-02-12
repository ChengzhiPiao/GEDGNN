from utils import tab_printer
from trainer import Trainer
from param_parser import parameter_parser

def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    tab_printer(args)
    trainer = Trainer(args)

    if args.model_epoch_start > 0:
        trainer.load(args.model_epoch_start)

    #trainer.cur_epoch = args.model_epoch_start
    #trainer.demo_score('test')

    if args.model_train == 1:
        for epoch in range(args.model_epoch_start, args.model_epoch_end):
            if epoch >= 3000 and epoch % 20 == 0:
                trainer.args.learning_rate *= 0.75
                with open(trainer.args.abs_path + trainer.args.result_path + 'results.txt', 'a') as f:
                    print("lr =", trainer.args.learning_rate, file=f)
            trainer.cur_epoch = epoch
            # trainer.gen_delta_graphs(epoch % 10)
            # trainer.init_graph_pairs()
            trainer.fit()
            if (epoch + 1) % 10 == 0:
                trainer.save(epoch + 1)
                trainer.score('test_large')
            # trainer.score('val')
            # trainer.score('test_small')
            # trainer.score('test_large')
            #if not args.demo:
                #trainer.score('test2')
    else:
        trainer.cur_epoch = args.model_epoch_start
        # trainer.gen_delta_graphs(9)
        # trainer.init_graph_pairs()
        # trainer.large_score('test_large', test_k=100)
        # trainer.score('test_large')

        """
        noah: test_k = 0
        greedy: tesk_k = 1, --greedy
        GedGNN: test_k = 1, 10 or 100
        """
        trainer.large_score('test_large', test_k=10)
        #for i in range(0, 5):
         #   trainer.large_score('test_large', gid=i, test_k=1)
        # trainer.large_score('test_large', gid=2, test_k=10)

        # trainer.batch_score('test', test_k=100)
        # trainer.batch_score('test_large', test_k=100)
        # trainer.batch_score('test_small', test_k=100)

        #trainer.score('test_large', test_k=0) # noah
        """
        test_matching = True
        trainer.cur_epoch = args.model_epoch_start
        #trainer.score('val', test_matching=test_matching)
        trainer.score('test', test_matching=test_matching)
        #if not args.demo:
         #   trainer.score('test2')
        """

if __name__ == "__main__":
    main()
