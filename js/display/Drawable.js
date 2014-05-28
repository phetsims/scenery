// Copyright 2002-2014, University of Colorado

/**
 * A unit that is drawable with a specific renderer.
 * NOTE: Drawables are assumed to be pooled with Poolable, as freeToPool() is called
 *
 * APIs for drawable types:
 *
 * DOM: {
 *   domElement: {HTMLElement}
 * }
 *
 * OHTWO TODO: add more API information, and update
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  
  var globalId = 1;
  
  scenery.Drawable = function Drawable( renderer ) {
    this.initializeDrawable( renderer );
  };
  var Drawable = scenery.Drawable;
  
  inherit( Object, Drawable, {
    initializeDrawable: function( renderer ) {
      // unique ID for drawables
      this.id = this.id || globalId++;
      
      sceneryLayerLog && sceneryLayerLog.Drawable && sceneryLayerLog.Drawable( '[' + this.constructor.name + '*] initialize ' + this.toString() );
      
      this.cleanDrawable();
      
      this.renderer = renderer;
      
      this.dirty = true;
      this.disposed = false;
      
      return this;
    },
    
    cleanDrawable: function() {
      // what drawble we are being rendered (or put) into (will be filled in later)
      this.parentDrawable = null;
      this.backbone = null; // a backbone reference (if applicable).
      
      // what our parent drawable will be after the stitch is finished
      this.pendingParentDrawable = null;
      this.pendingBackbone = null;
      
      // linked list handling (will be filled in later)
      this.previousDrawable = null;
      this.nextDrawable = null;
      
      // similar but pending handling, so that we can traverse both orders at the same time for stitching
      this.pendingPreviousDrawable = null;
      this.pendingNextDrawable = null;
    },
    
    setBlockBackbone: function( backboneInstance ) {
      this.parentDrawable = backboneInstance;
      this.backbone = backboneInstance;
      this.pendingParentDrawable = backboneInstance;
      this.pendingBackbone = backboneInstance;
    },
    
    setPendingBlock: function( block, backbone ) {
      assert && assert( backbone !== undefined, 'backbone can be either null or a backbone' );
      this.pendingParentDrawable = block;
      this.pendingBackbone = backbone;
    },
    
    removePendingBackbone: function( backbone ) {
      // Only update our pending information if it is still pointing to the backbone.
      // We want to ignore this call if our drawable has been set (pending) to another backbone (or no backbone at all, e.g. inline blocks)
      if ( backbone === this.pendingBackbone ) {
        this.pendingParentDrawable = null;
        this.pendingBackbone = null;
      }
    },
    
    updateBlock: function() {
      if ( this.parentDrawable !== this.pendingParentDrawable ) {
        this.parentDrawable && this.parentDrawable.removeDrawable( this );
        this.pendingParentDrawable && this.pendingParentDrawable.addDrawable( this );
        this.parentDrawable = this.pendingParentDrawable;
        this.backbone = this.pendingBackbone;
      }
    },
    
    markDirty: function() {
      if ( !this.dirty ) {
        this.dirty = true;
        
        // TODO: notify what we want to call repaint() later
        if ( this.parentDrawable ) {
          this.parentDrawable.markDirtyDrawable( this );
        }
      }
    },
    
    markForDisposal: function( display ) {
      display.markDrawableForDisposal( this );
    },
    
    dispose: function() {
      sceneryLayerLog && sceneryLayerLog.Drawable && sceneryLayerLog.Drawable( '[' + this.constructor.name + '*] dispose ' + this.toString() );
      
      this.cleanDrawable();
      this.disposed = true;
      
      // for now
      this.freeToPool();
    },
    
    toString: function() {
      return this.constructor.name + '#' + this.id;
    }
  } );
  
  return Drawable;
} );
