// Copyright 2018, University of Colorado Boulder

/**
 * "definition" type for generalized type of accessibility option. A "behavior function" takes in options, and mutates
 * them to achieve the correct accessible behavior for that node in the PDOM.
 *
 * REVIEW: There seem to be a lot of attributes that don't seem to be supported well for behavior functions, but I don't
 * REVIEW: see any documentation of it. I'd prefer that things would work, but I understand the challenges if we just
 * REVIEW: need to have things documented.
 * REVIEW: To reproduce an example where this leaves things in a bad state, I ran the scenery playground:
 * REVIEW:   window.assertions.disableAssert(); // phet.scenery usage breaks some cases, I just use this to work around.
 * REVIEW:   var n = new scenery.Node();
 * REVIEW:   n.tagName = 'p';
 * REVIEW:   n.ariaRole = 'example';
 * REVIEW:   n.helpTextBehavior = function( node, options, helpText ) {
 * REVIEW:     options.labelTagName = 'h1';
 * REVIEW:     options.labelContent = node.ariaRole; // NOTE! we depend on something that does NOT call onAccessibleContentChange
 * REVIEW:     return options;
 * REVIEW:   };
 * REVIEW:   scene.addChild( n );
 * REVIEW:   n.helpText = 'Bogus';
 * REVIEW:   n.ariaRole = 'this-should-change';
 * REVIEW: So according to our behavior, our labelContent should have changed to 'this-should-change' since it's
 * REVIEW: supposed to be a copy of the ariaRole (something I chose that does not call onAccessibleContentChange when
 * REVIEW: it updates). Instead it isn't updated immediately:
 * REVIEW: <div class="accessibility">
 * REVIEW:   <h1 tabindex="-1" id="label-2-11">example</h1>
 * REVIEW:   <p tabindex="-1" id="2-11" role="this-should-change"></p>
 * REVIEW: </div>
 * REVIEW: There are many cases where this "broken" behavior can happen. Notably, this can also happen in the REVERSE.
 * REVIEW: Notably, the "behavior" setup only overrides options WHEN things are rebuilt fully. When a peer is updated,
 * REVIEW: it ignores all behaviors (which is concerning!). Maybe we should be evaluating the behaviors on all updates.
 * REVIEW: Example (again in the Scenery playground):
 * REVIEW:   window.assertions.disableAssert(); // phet.scenery usage breaks some cases, I just use this to work around.
 * REVIEW:   var n = new scenery.Node()
 * REVIEW:   scene.addChild( n );
 * REVIEW:   n.tagName = 'p'
 * REVIEW:   n.helpText = 'Help text'; // using the "default" since the problem happens with it. SHOWS "Help text"
 * REVIEW:   n.descriptionContent = 'Oops'; // after this change, our behavior for helpText is not checked. SHOWS "Oops"
 * REVIEW:   n.tagName = 'b'; // some unrelated change that triggers rebuild. NOW it SHOWS "Help text" again since
 * REVIEW:                    // behaviors were checked/executed.
 * REVIEW: So to have some usages of setters called from update() where things can be overridden, and others called
 * REVIEW: directly WITHOUT behaviors seems to get into really buggy behavior.
 * //ZEPUMPH: Let's talk about this more as part of https://github.com/phetsims/scenery/issues/867
 *
 * REVIEW: I sorted through a lot of the logic, and here's my current assessment. "ok" basically means that the value
 * REVIEW: always gets set, and in a consistent way. So all of the "options" on the node that the update uses:
 * REVIEW:   tagName (ok)
 * REVIEW:   accessibleNamespace (ok)
 * REVIEW:   containerTagName (ok)
 * REVIEW:   containerAriaRole (setter doesn't call onAccessibleContentChange, but the setAccessibleAttribute/removeAccessibleAttribute probably work)
 * REVIEW:   labelTagName (ok)
 * REVIEW:   descriptionTagName (ok)
 * REVIEW:   appendLabel from orderElements (ok)
 * REVIEW:   appendDescription from orderElements (ok)
 * REVIEW:   labelContent (setter doesn't call onAccessibleContentChange, can introduce buggy behavior)
 * REVIEW:   innerContent (setter doesn't call onAccessibleContentChange, can introduce buggy behavior)
 * REVIEW:   descriptionContent (NOT included) -- check
 * REVIEW:   inputType (setter doesn't call onAccessibleContentChange, can introduce buggy behavior)
 * REVIEW:   ariaLabelledbyAssociations (ok, probably won't override with behaviors?)
 * REVIEW:   ariaDescribedbyAssociations (ok, probably won't override with behaviors?)
 * REVIEW:   accessibleAttributes (we WANT to override these with behavior, but can't right now) from onAttributeChange
 * REVIEW:   inputValue (doesn't look like behaviors can override)
 * REVIEW:   node.focusHighlight (looks like updates to this are completely broken)
 * REVIEW:   node.focusable (looks like updates to this are completely broken)
 * REVIEW: It would be nice to have all of these consistent, and have one path for setting if needed (that can both be
 * REVIEW: used by setters in Accessibility.js AND the "update()" for good performance).
 * //ZEPUMPH: Let's talk about this more as part of https://github.com/phetsims/scenery/issues/867
 *
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

define( function( require ) {
  'use strict';

  var scenery = require( 'SCENERY/scenery' );

  /**
   * @name a11yBehaviorFunction
   * A function that will mutate given options object to achieve the correct a11y structure in the PDOM.
   * @function
   * @param {Node} node - the node that the a11y behavior is being applied to
   * @param {Object} options - options to mutate within the function
   * @param {string} value - the value that you are setting the behavior of, like the accessibleName
   * @returns {Object} - the options that have been mutated by the behavior function.
   */
  var A11yBehaviorFunctionDef = {

    /**
     * Will assert out if the behavior function doesn't match the expected features of A11yBehaviorFunction
     * @param {function} behaviorFunction
     */
    validateA11yBehaviorFunctionDef: function( behaviorFunction ) {
      assert && assert( typeof behaviorFunction === 'function' );
      assert && assert( behaviorFunction.length === 3, 'behavior function should take three args' );
      assert && assert( typeof behaviorFunction( new scenery.Node(), {}, '' ) === 'object',
        'behavior function should return an object' );
    }
  };

  scenery.register( 'A11yBehaviorFunctionDef', A11yBehaviorFunctionDef );

  return A11yBehaviorFunctionDef;
} );
