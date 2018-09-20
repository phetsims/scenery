// Copyright 2015-2016, University of Colorado Boulder

/**
 * A sub-component of an Instance that handles matters relating to whether fitted blocks should not fit if possible.
 * We mostly mark our own drawables as fittable, and track whether our subtree is all fittable (so that common-ancestor
 * fits can determine if their bounds will change).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var Emitter = require( 'AXON/Emitter' );
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   * @constructor
   *
   * @param {Instance} instance - Our Instance, never changes.
   */
  function Fittability( instance ) {
    // @private {Instance}
    this.instance = instance;
  }

  scenery.register( 'Fittability', Fittability );

  inherit( Object, Fittability, {
    /**
     * Responsible for initialization and cleaning of this. If the parameters are both null, we'll want to clean our
     * external references (like Instance does).
     *
     * @param {Display|null} display
     * @param {Trail|null} trail
     * @returns {Fittability} - Returns this, to allow chaining.
     */
    initialize: function( display, trail ) {
      this.display = display; // @private {Display}
      this.trail = trail; // @private {Trail}
      this.node = trail && trail.lastNode(); // @private {Node}

      // @public {boolean} - When our instance's node has a particular combination of features and/or flags (like
      // preventDefault:true) that should make any FittedBlock containing drawables under that node OR that would
      // include the bounds of the node in its FittedBlock to not compute the relevant fit (e.g. have it take up the
      // full display instead).
      this.selfFittable = !!trail && this.isSelfFitSupported();

      // @public {boolean} - Whether this instance AND all of its ancestor instances (down to the root instance for the
      // display) all are self-fittable.
      this.ancestorsFittable = this.selfFittable;

      // @public {number} - The number of children whose subtrees have an unfittable instance, plus 1 if this instance
      // itself is unfittable. Using a number allows us to quickly increment/decrement when a particular child changes
      // its fittability (so we don't have to check other subtrees or traverse further up the tree). For a more
      // complete description of this technique, see RendererSummary.
      // This is important, since if it's 0, it indicates that this entire subtree has NO unfittable content. Thus if
      // a FittedBlock's common ancestor (for the common-ancestor fit) is this instance, we shouldn't have issues
      // updating our bounds.
      this.subtreeUnfittableCount = this.selfFittable ? 0 : 1;

      // @public {Emitter} - Called with no arguments when the subtree fittability changes (whether
      // subtreeUnfittableCount is greater than zero or not).
      this.subtreeFittabilityChange = this.subtreeFittabilityChange || new Emitter();

      return this; // allow chaining
    },

    /**
     * Easy access to our parent Instance's Fittability, if it exists.
     * @private
     *
     * @returns {Fittability|null}
     */
    get parent() {
      return this.instance.parent ? this.instance.parent.fittability : null;
    },

    /**
     * Called when the instance is updating its rendering state (as any fittability changes to existing instances will
     * trigger an update there).
     * @public
     */
    checkSelfFittability: function() {
      var newSelfFittable = this.isSelfFitSupported();
      if ( this.selfFittable !== newSelfFittable ) {
        this.updateSelfFittable();
      }
    },

    /**
     * Whether our node's performance flags allows the subtree to be fitted.
     * @private
     *
     * Any updates to flags (for instance, a 'dynamic' flag perhaps?) should be added here.
     *
     * @returns {boolean}
     */
    isSelfFitSupported: function() {
      return !this.node.isPreventFit();
    },

    /**
     * Called when our parent just became fittable. Responsible for flagging subtrees with the ancestorsFittable flag,
     * up to the point where they are fittable.
     * @private
     */
    markSubtreeFittable: function() {
      // Bail if we can't be fittable ourselves
      if ( !this.selfFittable ) {
        return;
      }

      this.ancestorsFittable = true;

      var children = this.instance.children;
      for ( var i = 0; i < children.length; i++ ) {
        children[ i ].fittability.markSubtreeFittable();
      }

      // Update the Instance's drawables, so that their blocks can potentially now be fitted.
      this.instance.updateDrawableFittability( true );
    },

    /**
     * Called when our parent just became unfittable and we are fittable. Responsible for flagging subtrees with
     * the !ancestorsFittable flag, up to the point where they are unfittable.
     * @private
     */
    markSubtreeUnfittable: function() {
      // Bail if we are already unfittable
      if ( !this.ancestorsFittable ) {
        return;
      }

      this.ancestorsFittable = false;

      var children = this.instance.children;
      for ( var i = 0; i < children.length; i++ ) {
        children[ i ].fittability.markSubtreeUnfittable();
      }

      // Update the Instance's drawables, so that their blocks can potentially now be prevented from being fitted.
      this.instance.updateDrawableFittability( false );
    },

    /**
     * Called when our Node's self fit-ability has changed.
     * @private
     */
    updateSelfFittable: function() {
      var newSelfFittable = this.isSelfFitSupported();
      assert && assert( this.selfFittable !== newSelfFittable );

      this.selfFittable = newSelfFittable;

      if ( this.selfFittable && ( !this.parent || this.parent.ancestorsFittable ) ) {
        this.markSubtreeFittable();
      }
      else if ( !this.selfFittable ) {
        this.markSubtreeUnfittable();
      }

      if ( this.selfFittable ) {
        this.decrementSubtreeUnfittableCount();
      }
      else {
        this.incrementSubtreeUnfittableCount();
      }
    },

    /**
     * A child instance's subtree became unfittable, OR our 'self' became unfittable. This is responsible for updating
     * the subtreeFittableCount for this instance AND up to all ancestors that would be affected by the change.
     * @private
     */
    incrementSubtreeUnfittableCount: function() {
      this.subtreeUnfittableCount++;

      // If now something in our subtree can't be fitted, we need to notify our parent
      if ( this.subtreeUnfittableCount === 1 ) {
        this.parent && this.parent.incrementSubtreeUnfittableCount();

        // Notify anything listening that the condition ( this.subtreeUnfittableCount > 0 ) changed.
        this.subtreeFittabilityChange.emit();
      }
    },

    /**
     * A child instance's subtree became fittable, OR our 'self' became fittable. This is responsible for updating
     * the subtreeFittableCount for this instance AND up to all ancestors that would be affected by the change.
     * @private
     */
    decrementSubtreeUnfittableCount: function() {
      this.subtreeUnfittableCount--;

      // If now our subtree can all be fitted, we need to notify our parent
      if ( this.subtreeUnfittableCount === 0 ) {
        this.parent && this.parent.decrementSubtreeUnfittableCount();

        // Notify anything listening that the condition ( this.subtreeUnfittableCount > 0 ) changed.
        this.subtreeFittabilityChange.emit();
      }
    },

    /**
     * Called when an instance is added as a child to our instance. Updates necessary counts.
     * @public
     *
     * @param {Fittability} childFittability - The Fittability of the new child instance.
     */
    onInsert: function( childFittability ) {
      if ( !this.ancestorsFittable ) {
        childFittability.markSubtreeUnfittable();
      }

      if ( childFittability.subtreeUnfittableCount > 0 ) {
        this.incrementSubtreeUnfittableCount();
      }
    },

    /**
     * Called when a child instance is removed from our instance. Updates necessary counts.
     * @public
     *
     * @param {Fittability} childFittability - The Fittability of the old child instance.
     */
    onRemove: function( childFittability ) {
      if ( !this.ancestorsFittable ) {
        childFittability.markSubtreeFittable();
      }

      if ( childFittability.subtreeUnfittableCount > 0 ) {
        this.decrementSubtreeUnfittableCount();
      }
    },

    /**
     * Sanity checks that run when slow assertions are enabled. Enforces the invariants of the Fittability subsystem.
     * @public
     */
    audit: function() {
      if ( assertSlow ) {
        assertSlow( this.selfFittable === this.isSelfFitSupported(),
          'selfFittable diverged from isSelfFitSupported()' );

        assertSlow( this.ancestorsFittable === ( ( this.parent ? this.parent.ancestorsFittable : true ) && this.selfFittable ),
          'Our ancestorsFittable should be false if our parent or our self is not fittable.' );

        // Our subtree unfittable count should be the sum of children that have a non-zero count, plus 1 if our self
        // is not fittable
        var subtreeUnfittableCount = 0;
        if ( !this.selfFittable ) {
          subtreeUnfittableCount++;
        }
        _.each( this.instance.children, function( instance ) {
          if ( instance.fittability.subtreeUnfittableCount > 0 ) {
            subtreeUnfittableCount++;
          }
        } );
        assertSlow( this.subtreeUnfittableCount === subtreeUnfittableCount, 'Incorrect subtreeUnfittableCount' );
      }
    }
  } );

  return Fittability;
} );
