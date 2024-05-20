* The calling process of TF concrete function
* 1. TF Wrap fucntion
ref. https://github.com/tensorflow/tensorflow/blob/216fce0329f8d92c11d1cf6ca67712f39432ddc6/tensorflow/python/saved_model/load.py
class _WrapperFunction(function.ConcreteFunction):
	def _call_flat(self, args, captured_inputs, cancellation_manager=None):
		return super(_WrapperFunction, self)._call_flat(args, captured_inputs,
                                                    cancellation_manager)
* 2. function.ConcreteFunction
ref. https://github.com/tensorflow/tensorflow/blob/216fce0329f8d92c11d1cf6ca67712f39432ddc6/tensorflow/python/eager/function.py#L1488
class ConcreteFunction(object):
  """Callable object encapsulating a function definition and its gradient.
	def __call__(self, *args, **kwargs):  # for Executing the wrapped function.
			....
    		# function_spec defines the structured signature.
     		self._set_function_spec(function_spec)
     			....
      		return self._call_impl(args, kwargs)
  	def _call_impl(self, args, kwargs):
      		return self._call_with_flat_signature(args, kwargs)
     	def _call_with_structured_signature(self, args, kwargs, cancellation_manager):
    		"""Executes the wrapped function with the structured signature.
         	return self._call_flat(
        		filtered_flat_args,
        		captured_inputs=self.captured_inputs,
        		cancellation_manager=cancellation_manager)
       def _call_flat(self, args, captured_inputs, cancellation_manager=None):
    		"""Executes the wrapped function.



