* The calling process of TF concrete function
* TF Wrap fucntion
ref. https://github.com/tensorflow/tensorflow/blob/216fce0329f8d92c11d1cf6ca67712f39432ddc6/tensorflow/python/saved_model/load.py
class _WrapperFunction(function.ConcreteFunction):
....
 def _call_flat(self, args, captured_inputs, cancellation_manager=None):
    return super(_WrapperFunction, self)._call_flat(args, captured_inputs,
                                                    cancellation_manager)
* function.ConcreteFunction
ref. https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/polymorphic_function/concrete_function.py#L1018
  def __call__(self, *args, **kwargs):  # for Executing the wrapped function.
      return self._call_impl(args, kwargs)
  def _call_impl(self, args, kwargs):
      return self._call_with_flat_signature(args, kwargs)
