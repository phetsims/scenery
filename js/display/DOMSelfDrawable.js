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
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  
  scenery.DOMSelfDrawable = function DOMSelfDrawable( trail, renderer, transformTrail, instance ) {
    Drawable.call( this, trail, renderer, transformTrail );
    this.instance = instance;
    
    this.node = instance.trail.lastNode();
    
    this.visualState = null; // to be created in initializeDOM
    this.dirty = true;
    this.domElement = this.node.initializeDOM( this );
    
    
    // TODO: handle transforms?
    // TODO: check for the "force acceleration" flag
  };
  var DOMSelfDrawable = scenery.DOMSelfDrawable;
  
  inherit( Drawable, DOMSelfDrawable, {
    markTransformDirty: function() {
      this.visualState.transformDirty = true;
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
    }
  } );
  
  return DOMSelfDrawable;
} );
