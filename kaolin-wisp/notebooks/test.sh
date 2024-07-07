for iter in {1..1}
do
    echo "Iteration $iter"
    python next_best_pose_range_new_aksh.py -i test/input_$iter.pkl -r /home/saptarshi/dev/CustomComposer/robot_range2_lc_0.04.pkl -p test/ransam500_out$iter.pkl -d test/ransam500_out$iter/ -a test/faltu >test/ransam500_out$iter.txt
    # python next_best_pose_range_new_aksh.py -i test/input_$iter.pkl -r /home/saptarshi/dev/CustomComposer/robot_range2_lc_0.04.pkl -p test/ransam500_nomove_out$iter.pkl  -d test/ransam500_nomove_out$iter/ -a test/faltu -l 0 >test/ransam500_nomove_out$iter.txt
done

