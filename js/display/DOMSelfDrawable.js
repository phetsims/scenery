// Copyright 2013-2016, University of Colorado Boulder

/**
 * DOM drawable for a single painted node.
 *
 * Subtypes should expose the following API that is used by DOMSelfDrawable:
 * - drawable.domElement {HTMLElement} - The primary DOM element that will get transformed and added.
 * - drawable.updateDOM() {function} - Called with no arguments in order to update the domElement's view.
 *
 * TODO: make abstract subtype methods for improved documentation
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );
  require( 'SCENERY/display/Renderer' );

  function DOMSelfDrawable( renderer, instance ) {
    this.initializeDOMSelfDrawable( renderer, instance );

    throw new Error( 'Should use initialization and pooling' );
  }

  scenery.register( 'DOMSelfDrawable', DOMSelfDrawable );

  inherit( SelfDrawable, DOMSelfDrawable, {
    initializeDOMSelfDrawable: function( renderer, instance ) {
      // this is the same across lifecycles
      this.transformListener = this.transformListener || this.markTransformDirty.bind( this );

      // super initialization
      this.initializeSelfDrawable( renderer, instance );

      this.forceAcceleration = false; // TODO: for now, check to see if this is used and how to use it
      this.markTransformDirty();

      this.visibilityDirty = true;

      // handle transform changes
      instance.relativeTransform.addListener( this.transformListener ); // when our relative tranform changes, notify us in the pre-repaint phase
      instance.relativeTransform.addPrecompute(); // trigger precomputation of the relative transform, since we will always need it when it is updated

      return this;
    },

    markTransformDirty: function() {
      // update the visual state available to updateDOM, so that it will update the transform (Text needs to change the transform, so it is included)
      this.transformDirty = true;

      this.markDirty();
    },

    // called from the Node, probably during updateDOM
    getTransformMatrix: function() {
      this.instance.relativeTransform.validate();
      return this.instance.relativeTransform.matrix;
    },

    /**
     * Updates the DOM appearance of this drawable (whether by preparing/calling draw calls, DOM element updates, etc.)
     * @public
     * @override
     *
     * @returns {boolean} - Whether the update should continue (if false, further updates in supertype steps should not
     *                      be done).
     */
    update: function() {
      // See if we need to actually update things (will bail out if we are not dirty, or if we've been disposed)
      if ( !SelfDrawable.prototype.update.call( this ) ) {
        return false;
      }

      this.updateDOM();

      if ( this.visibilityDirty ) {
        this.visibilityDirty = false;

        this.domElement.style.visibility = this.visible ? '' : 'hidden';
      }

      this.cleanPaintableState && this.cleanPaintableState();

      return true;
    },

    // @protected: called to update the visual appearance of our domElement
    updateDOM: function() {
      // should generally be overridden by drawable subtypes to implement the update
    },

    // @override
    updateSelfVisibility: function() {
      SelfDrawable.prototype.updateSelfVisibility.call( this );

      if ( !this.visibilityDirty ) {
        this.visibilityDirty = true;
        this.markDirty();
      }
    },

    dispose: function() {
      this.instance.relativeTransform.removeListener( this.transformListener );
      this.instance.relativeTransform.removePrecompute();

      // super call
      SelfDrawable.prototype.dispose.call( this );
    }
  } );

  return DOMSelfDrawable;
} );
