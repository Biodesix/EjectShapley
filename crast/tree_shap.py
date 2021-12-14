# This was grabbed from the shap package and altered to work on our data types

# extend our decision path with a fraction of one and zero extensions
#@numba.jit(nopython=True, nogil=True)
def extend_path(feature_indexes, zero_fractions, one_fractions, pweights,
                unique_depth, zero_fraction, one_fraction, feature_index):
    feature_indexes[unique_depth] = feature_index
    zero_fractions[unique_depth] = zero_fraction
    one_fractions[unique_depth] = one_fraction
    if unique_depth == 0:
        pweights[unique_depth] = 1.
    else:
        pweights[unique_depth] = 0.

    for i in range(unique_depth - 1, -1, -1):
        pweights[i + 1] += one_fraction * pweights[i] * (i + 1.) / (unique_depth + 1.)
        pweights[i] = zero_fraction * pweights[i] * (unique_depth - i) / (unique_depth + 1.)

# undo a previous extension of the decision path
#@numba.jit(nopython=True, nogil=True)
def unwind_path(feature_indexes, zero_fractions, one_fractions, pweights,
                unique_depth, path_index):
    one_fraction = one_fractions[path_index]
    zero_fraction = zero_fractions[path_index]
    next_one_portion = pweights[unique_depth]

    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0.:
            tmp = pweights[i]
            pweights[i] = next_one_portion * (unique_depth + 1.) / ((i + 1.) * one_fraction)
            next_one_portion = tmp - pweights[i] * zero_fraction * (unique_depth - i) / (unique_depth + 1.)
        else:
            pweights[i] = (pweights[i] * (unique_depth + 1)) / (zero_fraction * (unique_depth - i))

    for i in range(path_index, unique_depth):
        feature_indexes[i] = feature_indexes[i + 1]
        zero_fractions[i] = zero_fractions[i + 1]
        one_fractions[i] = one_fractions[i + 1]

# determine what the total permuation weight would be if
# we unwound a previous extension in the decision path
#@numba.jit(nopython=True, nogil=True)
def unwound_path_sum(feature_indexes, zero_fractions, one_fractions, pweights, unique_depth, path_index):
    one_fraction = one_fractions[path_index]
    zero_fraction = zero_fractions[path_index]
    next_one_portion = pweights[unique_depth]
    total = 0

    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0.:
            tmp = next_one_portion * (unique_depth + 1.) / ((i + 1.) * one_fraction)
            total += tmp
            next_one_portion = pweights[i] - tmp * zero_fraction * ((unique_depth - i) / (unique_depth + 1.))
        else:
            total += (pweights[i] / zero_fraction) / ((unique_depth - i) / (unique_depth + 1.))

    return total

def tree_shap_recursive(children_left, children_right, children_default, features, thresholds, values, node_sample_weight,
                        x, x_missing, phi, node_index, unique_depth, parent_feature_indexes,
                        parent_zero_fractions, parent_one_fractions, parent_pweights, parent_zero_fraction,
                        parent_one_fraction, parent_feature_index, condition, condition_feature, condition_fraction):

    # stop if we have no weight coming down to us
    if condition_fraction == 0.:
        return

    # extend the unique path
    feature_indexes = parent_feature_indexes[unique_depth + 1:]
    feature_indexes[:unique_depth + 1] = parent_feature_indexes[:unique_depth + 1]
    zero_fractions = parent_zero_fractions[unique_depth + 1:]
    zero_fractions[:unique_depth + 1] = parent_zero_fractions[:unique_depth + 1]
    one_fractions = parent_one_fractions[unique_depth + 1:]
    one_fractions[:unique_depth + 1] = parent_one_fractions[:unique_depth + 1]
    pweights = parent_pweights[unique_depth + 1:]
    pweights[:unique_depth + 1] = parent_pweights[:unique_depth + 1]

    if condition == 0 or condition_feature != parent_feature_index:
        extend_path(
            feature_indexes, zero_fractions, one_fractions, pweights,
            unique_depth, parent_zero_fraction, parent_one_fraction, parent_feature_index
        )

    split_index = features[node_index]

    # leaf node
    if children_right[node_index] < 0:
        for i in range(1, unique_depth+1):
            w = unwound_path_sum(feature_indexes, zero_fractions, one_fractions, pweights, unique_depth, i)
            # phi[feature_indexes[i]] += w * (one_fractions[i] - zero_fractions[i]) * values[node_index] * condition_fraction
            phi[feature_indexes[i],:] += w * (one_fractions[i] - zero_fractions[i]) * values[node_index] * condition_fraction
    # internal node
    else:
        # find which branch is "hot" (meaning x would follow it)
        hot_index = 0
        cleft = children_left[node_index]
        cright = children_right[node_index]
        if x_missing[split_index] == 1:
            hot_index = children_default[node_index]
        elif x[split_index] < thresholds[node_index]:
            hot_index = cleft
        else:
            hot_index = cright
        cold_index = (cright if hot_index == cleft else cleft)
        w = node_sample_weight[node_index]
        hot_zero_fraction = node_sample_weight[hot_index] / w
        cold_zero_fraction = node_sample_weight[cold_index] / w
        incoming_zero_fraction = 1.
        incoming_one_fraction = 1.

        # see if we have already split on this feature,
        # if so we undo that split so we can redo it for this node
        path_index = 0
        while (path_index <= unique_depth):
            if feature_indexes[path_index] == split_index:
                break
            path_index += 1

        if path_index != unique_depth + 1:
            incoming_zero_fraction = zero_fractions[path_index]
            incoming_one_fraction = one_fractions[path_index]
            unwind_path(feature_indexes, zero_fractions, one_fractions, pweights, unique_depth, path_index)
            unique_depth -= 1

        # divide up the condition_fraction among the recursive calls
        hot_condition_fraction = condition_fraction
        cold_condition_fraction = condition_fraction
        if condition > 0 and split_index == condition_feature:
            cold_condition_fraction = 0.
            unique_depth -= 1
        elif condition < 0 and split_index == condition_feature:
            hot_condition_fraction *= hot_zero_fraction
            cold_condition_fraction *= cold_zero_fraction
            unique_depth -= 1

        tree_shap_recursive(
            children_left, children_right, children_default, features, thresholds, values, node_sample_weight,
            x, x_missing, phi, hot_index, unique_depth + 1,
            feature_indexes, zero_fractions, one_fractions, pweights,
            hot_zero_fraction * incoming_zero_fraction, incoming_one_fraction,
            split_index, condition, condition_feature, hot_condition_fraction
        )

        tree_shap_recursive(
            children_left, children_right, children_default, features, thresholds, values, node_sample_weight,
            x, x_missing, phi, cold_index, unique_depth + 1,
            feature_indexes, zero_fractions, one_fractions, pweights,
            cold_zero_fraction * incoming_zero_fraction, 0,
            split_index, condition, condition_feature, cold_condition_fraction
        )

