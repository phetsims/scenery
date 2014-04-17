// Copyright 2002-2013, University of Colorado

/**
 * DOM drawable for a specific painted node. TODO docs
 *
 * Node API needed:
 * {
 *   attachDOMDrawable: function( DOMSelfDrawable ) : DOMElement    Node itself responsible for pooling available DOM elements and state. Can use
                                                                drawable.visualState to set flags and state.
 *   detachDOMDrawable: function( DOMSelfDrawable )
 * }
 *
 * visual state API needed:
 * {
 *   markPaintDirty: function()  // used by Strokable/Fillable
 *   node: Node                  // used by Strokable/Fillable
 *   transformDirty: Boolean     // set by the drawable
 *   forceAcceleration: Boolean  // set by the drawable. for now, won't change once created
 *   drawable: DOMSelfDrawable   // set by the visual state on initialization     NOTE: required for any type of visual state! (used by Strokable/Fillable states)
 *   domElement: DOMElement      // what we use to render the DOM element (should be the base of the displayed element)
 *   updateDOM: function()       // updates any visual state, including the transform
 *   onDetach: function()        // called when the state is detached from a drawable. optionally discard DOM elements. we guarantee state will be
 *                               // initialized again before any more update() calls
 * }
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  require( 'SCENERY/layers/Renderer' );
  
  scenery.DOMSelfDrawable = function DOMSelfDrawable( renderer, instance ) {
    Drawable.call( this, renderer );
    this.instance = instance;
    
    this.node = instance.trail.lastNode();
    
    this.visualState = null; // to be created in attachDOMDrawable
    this.dirty = true;
    
    this.node.attachDOMDrawable( this ); // should set this.visualState
    
    // now that we called attachDOMDrawable, update the visualState object with the flags it will need
    this.domElement = this.visualState.domElement;
    this.visualState.forceAcceleration = ( renderer & scenery.Renderer.bitmaskForceAcceleration ) !== 0;
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
    
    // called from the Node, probably during updateDOM
    getTransformMatrix: function() {
      this.instance.validateRelativeTransform();
      return this.instance.relativeMatrix;
    },
    
    // called from elsewhere to update the DOM element
    repaint: function() {
      if ( this.dirty ) {
        this.dirty = false;
        this.visualState.updateDOM();
      }
    },
    
    dispose: function() {
      // super call
      Drawable.prototype.dispose.call( this );
      
      this.node.detachDOMDrawable( this );
      
      this.instance.removeRelativeTransformListener( this.transformListener );
      this.instance.removeRelativeTransformPrecompute();
    }
  } );
  
  return DOMSelfDrawable;
} );
