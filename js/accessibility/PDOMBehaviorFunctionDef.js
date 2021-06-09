// Copyright 2018-2021, University of Colorado Boulder

/**
 * "definition" type for generalized type of accessibility option. A "behavior function" takes in options, and mutates
 * them to achieve the correct accessible behavior for that node in the PDOM.
 *
 * This type also holds many constant behavior functions for achieving certain structures in the PDOM using the "higher
 * level" API.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import scenery from '../scenery.js';

/**
 * @name pdomBehaviorFunction
 * A function that will mutate given options object to achieve the correct structure in the PDOM.
 * @function
 * @param {Node} node - the node that the pdom behavior is being applied to
 * @param {Object} [options] - options to mutate within the function
 * @param {string} value - the value that you are setting the behavior of, like the accessibleName
 * @param {function[]} callbacksForOtherNodes - behavior function also support taking state from a Node and using it to
 * set the accessible content for another Node. If this is the case, that logic should be set in a closure and added to
 * this list for execution after this Node is fully created. See discussion in https://github.com/phetsims/sun/issues/503#issuecomment-676541373
 * @returns {Object} - the options that have been mutated by the behavior function.
 */
const PDOMBehaviorFunctionDef = {

  /**
   * Will assert out if the behavior function doesn't match the expected features of pdomBehaviorFunction
   * @param {function} behaviorFunction
   */
  validatePDOMBehaviorFunctionDef( behaviorFunction ) {
    assert && assert( typeof behaviorFunction === 'function' );
    assert && assert( behaviorFunction.length === 3 || behaviorFunction.length === 4, 'behavior function should take three or four args' );
  }
};

scenery.register( 'PDOMBehaviorFunctionDef', PDOMBehaviorFunctionDef );

export default PDOMBehaviorFunctionDef;