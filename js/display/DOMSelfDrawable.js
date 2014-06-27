// Copyright 2002-2014, University of Colorado

/**
 * DOM drawable for a single painted node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );
  require( 'SCENERY/display/Renderer' );

  scenery.DOMSelfDrawable = function DOMSelfDrawable( renderer, instance ) {
    this.initializeDOMSelfDrawable( renderer, instance );

    throw new Error( 'Should use initialization and pooling' );
  };
  var DOMSelfDrawable = scenery.DOMSelfDrawable;

  inherit( SelfDrawable, DOMSelfDrawable, {
    initializeDOMSelfDrawable: function( renderer, instance ) {
      // this is the same across lifecycles
      this.transformListener = this.transformListener || this.markTransformDirty.bind( this );

      // super initialization
      this.initializeSelfDrawable( renderer, instance );

      this.forceAcceleration = ( renderer & scenery.Renderer.bitmaskForceAcceleration ) !== 0;
      this.markTransformDirty();

      // handle transform changes
      instance.addRelativeTransformListener( this.transformListener ); // when our relative tranform changes, notify us in the pre-repaint phase
      instance.addRelativeTransformPrecompute(); // trigger precomputation of the relative transform, since we will always need it when it is updated

      return this;
    },

    markTransformDirty: function() {
      // update the visual state available to updateDOM, so that it will update the transform (Text needs to change the transform, so it is included)
      this.transformDirty = true;

      this.markDirty();
    },

    // called from the Node, probably during updateDOM
    getTransformMatrix: function() {
      this.instance.validateRelativeTransform();
      return this.instance.relativeMatrix;
    },

    // called from elsewhere to update the DOM element
    update: function() {
      if ( this.dirty ) {
        this.dirty = false;
        this.updateDOM();
      }
    },

    // @protected: called to update the visual appearance of our domElement
    updateDOM: function() {
      // should generally be overridden by drawable subtypes to implement the update
    },

    onAttach: function( node ) {

    },

    onDetach: function( node ) {
      //OHTWO TODO: are we missing the disposal?
    },

    dispose: function() {
      this.instance.removeRelativeTransformListener( this.transformListener );
      this.instance.removeRelativeTransformPrecompute();

      // super call
      SelfDrawable.prototype.dispose.call( this );
    }
  } );

  return DOMSelfDrawable;
} );
