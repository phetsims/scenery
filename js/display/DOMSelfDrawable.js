// Copyright 2002-2013, University of Colorado

/**
 * DOM drawable for a specific painted node. TODO docs
 *
 * API needed:
 * {
 *   initializeDOM: function( DOMSelfDrawable ) : DOMElement    Node itself responsible for pooling available DOM elements and state. Can use drawable.visualState to set flags and state
 *   updateDOM: function( DOMSelfDrawable )
 *   destroyDOM: function( DOMSelfDrawable )
 * }
 *
 * drawable.visualState.transformDirty should be included, and can be set to true by the drawable
 * drawable.visualState.forceAcceleration should be included, and can be set to true by the drawable
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  
  scenery.DOMSelfDrawable = function DOMSelfDrawable( trail, renderer, instance ) {
    Drawable.call( this, trail, renderer );
    this.instance = instance;
    
    this.node = instance.trail.lastNode();
    
    this.visualState = null; // to be created in initializeDOM
    this.dirty = true;
    
    this.domElement = this.node.initializeDOM( this );
    
    // now that we called initializeDOM, update the visualState object with the flags it will need
    this.visualState.forceAcceleration = renderer & bitmaskForceAcceleration !== 0;
    this.markTransformDirty();
    
    // handle transform changes
    this.transformListener = this.markTransformDirty.bind( this );
    this.instance.addRelativeTransformListener( this.transformListener ); // when our relative tranform changes, notify us in the pre-repaint phase
    this.instance.addRelativeTransformPrecompute(); // trigger precomputation of the relative transform, since we will always need it when it is updated
  };
  var DOMSelfDrawable = scenery.DOMSelfDrawable;
  
  inherit( Drawable, DOMSelfDrawable, {
    markTransformDirty: function() {
      // update the visual state available to updateDOM, so that it will update the transform (Text needs to change the transform, so it is included)
      this.visualState.transformDirty = true;
      
      this.markDirty();
    },
    
    // called from the Node that we called initializeDOM on. should never be called after destroyDOM.
    markDirty: function() {
      if ( !this.dirty ) {
        this.dirty = true;
        
        // TODO: notify what we want to call update() later
        if ( this.block ) {
          this.block.markDOMDirty( this );
        }
      }
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
        this.node.updateDOM( this );
      }
    },
    
    dispose: function() {
      // super call
      Drawable.prototype.dispose.call( this );
      
      this.node.destroyDOM( this );
      
      this.instance.removeRelativeTransformListener( this.transformListener );
      this.instance.removeRelativeTransformPrecompute();
    }
  } );
  
  return DOMSelfDrawable;
} );
