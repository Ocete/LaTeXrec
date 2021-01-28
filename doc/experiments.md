# List of experiments run

### Toy dataset

All of these remove ambiguities since our toy dataset doesn't have any ambiguities.

- Vanilla: https://console.paperspace.com/latexrec/projects/prqyjc7d5/experiments/e32etyfasib6s/jobs/jvnk73evoq69q/logs

- Resnet instead of vanilla: https://console.paperspace.com/latexrec/projects/prqyjc7d5/experiments/e6msm5awdgq6a/jobs/jsnc7xsv34ibn/logs

- Vanilla using performer-attention-encoder: https://console.paperspace.com/latexrec/projects/prqyjc7d5/experiments/efn478hidhyvq/jobs/js9j32j4uxunv2/logs

### Im2latex dataset interesting / useful:

80k sample - 10 epochs: https://console.paperspace.com/latexrec/projects/prqyjc7d5/experiments/eswuqaxbs4ogfs/jobs/j9smguji0xlnm/logs

40 hours - stopped (ResNet):
https://console.paperspace.com/latexrec/projects/prqyjc7d5/experiments/e9fckpef6885y/jobs/jsinekwgd39lk7/logs

Different learning rate behaviour: https://console.paperspace.com/latexrec/projects/prqyjc7d5/experiments/es1ewlipk5alk1/jobs/jcto3xub2ig9i/logs


### Im2latex first official day (wednesday night)

All of these experiments run 3 epochs and early stopping with 0.005 min val_acc increment in 5 evaluations unless stated otherwise.

1. Vanilla + no remove_ambiguities + no other options: https://console.paperspace.com/latexrec/projects/prqyjc7d5/experiments/esm3wd5ct5wznb/jobs/jsjmbmp1nmkgkr/logs

2. Vanila + WITH remove_ambiguities + no other options: https://console.paperspace.com/latexrec/projects/prqyjc7d5/experiments/ev0kliwo7ig7n/jobs/js4yitc5qd5y3f/logs

(At this point in time we reason why remove_ambiguities is awesome and use it everywhere)

In parallel:

3a. Test Fast Attention: https://console.paperspace.com/latexrec/projects/prqyjc7d5/experiments/eaipoouvxu51/jobs/jsbacz6zazoftx/logs

3b. Test Resnet: https://console.paperspace.com/latexrec/projects/prqyjc7d5/experiments/er44mwm6f93lm

3c. Test 2d positional encoding: https://console.paperspace.com/latexrec/projects/prqyjc7d5/experiments/essvdxud3iune6/jobs/jsaibfq0sjoa57/logs

For these 2 experiments we reduce min val_acc increment to 0.0001 in 10 evaluations, and epochs to 5:

- Big boy: Fast Attention + ResNet + Positional Encoding 2d: https://console.paperspace.com/latexrec/projects/prqyjc7d5/experiments/es1xpacplbe75a/jobs/jglgq6cp4q0qx/logs

- Bigger boy: num_layers 1->4, num_heads 1->4, depth 16->32: https://console.paperspace.com/latexrec/projects/prqyjc7d5/experiments/esysqdw6apzohk

### Re-doing small experiments with more freer params for early stopping

All of these experiments run 2 epochs and early stopping with 0.001 min val_acc increment in 10 evaluations unless stated otherwise.

1. https://console.paperspace.com/latexrec/projects/prqyjc7d5/experiments/emkbizse5xbwb/jobs/jx9tfvqz99kfr/logs

2. https://console.paperspace.com/latexrec/projects/prqyjc7d5/experiments/es2853qhhkjz55/jobs/jx5l642bn7z03/logs

3a. https://console.paperspace.com/latexrec/projects/prqyjc7d5/experiments/es17uxobj51py9/jobs/jsw1sizw1ufizl/logs

3b. https://console.paperspace.com/latexrec/projects/prqyjc7d5/experiments/est5b43tpjdn4t/jobs/jkfycl2xm3jty/logs

3c. https://console.paperspace.com/latexrec/projects/prqyjc7d5/experiments/e8j2lchkw4ow4/jobs/jsizzvhvqsqvrg/logs