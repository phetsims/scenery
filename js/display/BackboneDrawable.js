// Copyright 2002-2014, University of Colorado

/**
 * A "backbone" block that controls a DOM element (usually a div) that contains other blocks with DOM/SVG/Canvas/WebGL content
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var CanvasBlock = require( 'SCENERY/display/CanvasBlock' );
  var SVGBlock = require( 'SCENERY/display/SVGBlock' );
  var DOMBlock = require( 'SCENERY/display/DOMBlock' );
  var Util = require( 'SCENERY/util/Util' );
  
  scenery.BackboneDrawable = function BackboneDrawable( display, backboneInstance, transformRootInstance, renderer, isDisplayRoot ) {
    this.initialize( display, backboneInstance, transformRootInstance, renderer, isDisplayRoot );
  };
  var BackboneDrawable = scenery.BackboneDrawable;
  
  inherit( Drawable, BackboneDrawable, {
    initialize: function( display, backboneInstance, transformRootInstance, renderer, isDisplayRoot ) {
      Drawable.call( this, renderer );
      
      this.display = display;
      
      this.forceAcceleration = Renderer.isAccelerationForced( this.renderer );
      
      // reference to the instance that controls this backbone
      this.backboneInstance = backboneInstance;
      
      // where is the transform root for our generated blocks?
      this.transformRootInstance = transformRootInstance;
      
      // where have filters been applied to up? our responsibility is to apply filters between this and our backboneInstance
      this.filterRootAncestorInstance = backboneInstance.parent ? backboneInstance.parent.getFilterRootInstance() : backboneInstance;
      
      // where have transforms been applied up to? our responsibility is to apply transforms between this and our backboneInstance
      this.transformRootAncestorInstance = backboneInstance.parent ? backboneInstance.parent.getTransformRootInstance() : backboneInstance;
      
      this.willApplyTransform = this.transformRootAncestorInstance !== this.transformRootInstance;
      this.willApplyFilters = this.filterRootAncestorInstance !== this.backboneInstance;
      
      this.transformListener = this.transformListener || this.markTransformDirty.bind( this );
      if ( this.willApplyTransform ) {
        this.backboneInstance.addRelativeTransformListener( this.transformListener ); // when our relative tranform changes, notify us in the pre-repaint phase
        this.backboneInstance.addRelativeTransformPrecompute(); // trigger precomputation of the relative transform, since we will always need it when it is updated
      }
      
      this.renderer = renderer;
      this.domElement = isDisplayRoot ? display._domElement : BackboneDrawable.createDivBackbone();
      this.isDisplayRoot = isDisplayRoot;
      this.dirtyDrawables = cleanArray( this.dirtyDrawables );
      
      Util.prepareForTransform( this.domElement, this.forceAcceleration );
      
      // if we need to, watch nodes below us (and including us) and apply their filters (opacity/visibility/clip) to the backbone.
      this.watchedFilterNodes = cleanArray( this.watchedFilterNodes );
      this.opacityDirty = true;
      this.visibilityDirty = true;
      this.clipDirty = true;
      this.opacityDirtyListener = this.opacityDirtyListener || this.markOpacityDirty.bind( this );
      this.visibilityDirtyListener = this.visibilityDirtyListener || this.markVisibilityDirty.bind( this );
      this.clipDirtyListener = this.clipDirtyListener || this.markClipDirty.bind( this );
      if ( this.willApplyFilters ) {
        assert && assert( this.filterRootAncestorInstance.trail.nodes.length < this.backboneInstance.trail.nodes.length,
                          'Our backboneInstance should be deeper if we are applying filters' );
        
        // walk through to see which instances we'll need to watch for filter changes
        for ( var instance = this.backboneInstance; instance !== this.filterRootAncestorInstance; instance = instance.parent ) {
          var node = instance.node;
          
          this.watchedFilterNodes.push( node );
          node.onStatic( 'opacity', this.opacityDirtyListener );
          node.onStatic( 'visibility', this.visibilityDirtyListener );
          node.onStatic( 'clip', this.clipDirtyListener );
        }
      }
      
      this.lastZIndex = 0; // our last zIndex is stored, so that overlays can be added easily
      
      this.blocks = this.blocks || []; // we are responsible for their disposal
      
      //OHTWO @deprecated
      this.lastFirstDrawable = null;
      this.lastLastDrawable = null;
      
      // We track whether our drawables were marked for removal (in which case, they should all be removed by the time we dispose).
      // If removedDrawables = false during disposal, it means we need to remove the drawables manually (this should only happen if an instance tree is removed)
      this.removedDrawables = false;
      
      sceneryLog && sceneryLog.BackboneDrawable && sceneryLog.BackboneDrawable( 'initialized ' + this.toString() );
      
      return this; // chaining
    },
    
    dispose: function() {
      sceneryLog && sceneryLog.BackboneDrawable && sceneryLog.BackboneDrawable( 'dispose ' + this.toString() );
      
      while ( this.watchedFilterNodes.length ) {
        var node = this.watchedFilterNodes.pop();
        
        node.offStatic( 'opacity', this.opacityDirtyListener );
        node.offStatic( 'visibility', this.visibilityDirtyListener );
        node.offStatic( 'clip', this.clipDirtyListener );
      }
      
      // if we need to remove drawables from the blocks, do so
      if ( !this.removedDrawables ) {
        for ( var d = this.lastFirstDrawable; d !== null; d = d.nextDrawable ) {
          d.parentDrawable.removeDrawable( d );
          if ( d === this.lastLastDrawable ) { break }
        }
      }
      
      this.markBlocksForDisposal();
      
      if ( this.willApplyTransform ) {
        this.backboneInstance.removeRelativeTransformListener( this.transformListener );
        this.backboneInstance.removeRelativeTransformPrecompute();
      }
      
      this.backboneInstance = null;
      this.transformRootInstance = null;
      this.filterRootAncestorInstance = null;
      this.transformRootAncestorInstance = null;
      cleanArray( this.dirtyDrawables );
      cleanArray( this.watchedFilterNodes );
      
      Drawable.prototype.dispose.call( this );
    },
    
    // dispose all of the blocks while clearing our references to them
    markBlocksForDisposal: function() {
      while ( this.blocks.length ) {
        var block = this.blocks.pop();
        this.domElement.removeChild( block.domElement );
        block.markForDisposal( this.display );
      }
    },
    
    // should be called during syncTree
    markForDisposal: function( display ) {
      for ( var d = this.lastFirstDrawable; d !== null; d = d.nextDrawable ) {
        d.removePendingBackbone( this );
        this.display.markDrawableChangedBlock( d );
        if ( d === this.lastLastDrawable ) { break }
      }
      this.removedDrawables = true;
      
      // super call
      Drawable.prototype.markForDisposal.call( this, display );
    },
    
    markDirtyDrawable: function( drawable ) {
      this.dirtyDrawables.push( drawable );
      this.markDirty();
    },
    
    markTransformDirty: function() {
      assert && assert( this.willApplyTransform, 'Sanity check for willApplyTransform' );
      
      // relative matrix on backbone instance should be up to date, since we added the compute flags
      scenery.Util.applyPreparedTransform( this.backboneInstance.relativeMatrix, this.domElement, this.forceAcceleration );
    },
    
    markOpacityDirty: function() {
      if ( !this.opacityDirty ) {
        this.opacityDirty = true;
        this.markDirty();
      }
    },
    
    markVisibilityDirty: function() {
      if ( !this.visibilityDirty ) {
        this.visibilityDirty = true;
        this.markDirty();
      }
    },
    
    markClipDirty: function() {
      if ( !this.clipDirty ) {
        this.clipDirty = true;
        this.markDirty();
      }
    },
    
    update: function() {
      if ( this.dirty && !this.disposed ) {
        this.dirty = false;
        
        while ( this.dirtyDrawables.length ) {
          this.dirtyDrawables.pop().update();
        }
        
        if ( this.opacityDirty ) {
          this.opacityDirty = false;
          
          var filterOpacity = this.willApplyFilters ? this.getFilterOpacity() : 1;
          this.domElement.style.opacity = ( filterOpacity !== 1 ) ? filterOpacity : '';
        }
        
        if ( this.visibilityDirty ) {
          this.visibilityDirty = false;
          
          this.domElement.style.display = ( this.willApplyFilters && !this.getFilterVisibility() ) ? 'none' : '';
        }
        
        if ( this.clipDirty ) {
          this.clipDirty = false;
          
          var clip = this.willApplyFilters ? this.getFilterClip() : '';
          
          //OHTWO TODO: CSS clip-path/mask support here. see http://www.html5rocks.com/en/tutorials/masking/adobe/
          this.domElement.style.clipPath = clip; // yikes! temporary, since we already threw something?
        }
      }
    },
    
    getFilterOpacity: function() {
      var opacity = 1;
      
      var len = this.watchedFilterNodes.length;
      for ( var i = 0; i < len; i++ ) {
        opacity *= this.watchedFilterNodes[i].getOpacity();
      }
      
      return opacity;
    },
    
    getFilterVisibility: function() {
      var len = this.watchedFilterNodes.length;
      for ( var i = 0; i < len; i++ ) {
        if ( !this.watchedFilterNodes[i].isVisible() ) {
          return false;
        }
      }
      
      return true;
    },
    
    getFilterClip: function() {
      var clip = '';
      
      var len = this.watchedFilterNodes.length;
      for ( var i = 0; i < len; i++ ) {
        if ( this.watchedFilterNodes[i]._clipArea ) {
          throw new Error( 'clip-path for backbones unimplemented, and with questionable browser support!' );
        }
      }
      
      return clip;
    },
    
    stitch: function( firstDrawable, lastDrawable, oldDrawableBeforeChange, oldDrawableAfterChange ) {
      sceneryLog && sceneryLog.BackboneDrawable && sceneryLog.BackboneDrawable( 'stitch ' + this.toString() + ' first:' + ( firstDrawable ? firstDrawable.toString() : 'null' ) + ' last:' + ( lastDrawable ? lastDrawable.toString() : 'null' ) );
      sceneryLog && sceneryLog.BackboneDrawable && sceneryLog.push();
      
      for ( var d = this.lastFirstDrawable; d !== null; d = d.nextDrawable ) {
        d.removePendingBackbone( this );
        this.display.markDrawableChangedBlock( d );
        if ( d === this.lastLastDrawable ) { break }
      }
      
      this.lastFirstDrawable = firstDrawable;
      this.lastLastDrawable = lastDrawable;
      
      this.markBlocksForDisposal();
      
      var currentBlock = null;
      var currentRenderer = 0;
      var firstDrawableForBlock = null;
      
      // linked-list iteration inclusively from firstDrawable to lastDrawable
      for ( var drawable = firstDrawable; drawable !== null; drawable = drawable.pendingNextDrawable || drawable.nextDrawable ) {
        
        // if we need to switch to a new block, create it
        if ( !currentBlock || drawable.renderer !== currentRenderer ) {
          if ( currentBlock ) {
            currentBlock.notifyInterval( firstDrawableForBlock, drawable.pendingPreviousDrawable );
          }
          
          currentRenderer = drawable.renderer;
          
          if ( Renderer.isCanvas( currentRenderer ) ) {
            currentBlock = CanvasBlock.createFromPool( this.display, currentRenderer, this.transformRootInstance );
          } else if ( Renderer.isSVG( currentRenderer ) ) {
            //OHTWO TODO: handle filter root separately from the backbone instance?
            currentBlock = SVGBlock.createFromPool( this.display, currentRenderer, this.transformRootInstance, this.backboneInstance );
          } else if ( Renderer.isDOM( currentRenderer ) ) {
            currentBlock = DOMBlock.createFromPool( this.display, drawable );
            currentRenderer = 0; // force a new block for the next drawable
          } else {
            throw new Error( 'unsupported renderer for BackboneDrawable.rebuild: ' + currentRenderer );
          }
          
          this.blocks.push( currentBlock );
          currentBlock.setBlockBackbone( this );
          this.domElement.appendChild( currentBlock.domElement ); //OHTWO TODO: minor speedup by appending only once its fragment is constructed? or use DocumentFragment?
          
          // mark it dirty for now, so we can check
          this.markDirtyDrawable( currentBlock );
          
          firstDrawableForBlock = drawable;
        }
        
        drawable.setPendingBlock( currentBlock, this );
        this.display.markDrawableChangedBlock( drawable );
        
        // pending linked list => linked list
        //OHTWO TODO: scan these after all backbones have been stitched! EEK!
        // throw new Error( 'see comment' );
        if ( drawable.pendingPreviousDrawable !== null ) {
          drawable.pendingPreviousDrawable.pendingNextDrawable = null;
          drawable.pendingPreviousDrawable.nextDrawable = drawable;
          drawable.previousDrawable = drawable.pendingPreviousDrawable;
          drawable.pendingPreviousDrawable = null;
        } else {
          drawable.previousDrawable = null;
        }
        
        // don't cause an infinite loop!
        if ( drawable === lastDrawable ) { break }
      }
      if ( lastDrawable ) {
        lastDrawable.pendingNextDrawable = null;
        lastDrawable.nextDrawable = null;
      }
      if ( currentBlock ) {
        currentBlock.notifyInterval( firstDrawableForBlock, lastDrawable );
      }
      
      // full-pass change for zindex. OHTWO TODO: only change where necessary
      var zIndex = 1; // don't start below 1
      for ( var k = 0; k < this.blocks.length; k++ ) {
        this.blocks[k].domElement.style.zIndex = zIndex++; // NOTE: this should give it its own stacking index (which is what we want)
      }
      
      this.lastZIndex = zIndex;
      
      sceneryLog && sceneryLog.BackboneDrawable && sceneryLog.pop();
    },
    
    audit: function( allowPendingBlock, allowPendingList, allowDirty ) {
      if ( assertSlow ) {
        Drawable.prototype.audit.call( this, allowPendingBlock, allowPendingList, allowDirty );
        
        assertSlow && assertSlow( this.backboneInstance.state.isBackbone, 'We should reference an instance that requires a backbone' );
        assertSlow && assertSlow( this.transformRootInstance.state.isTransformed, 'Transform root should be transformed' );
        
        for ( var i = 0; i < this.blocks.length; i++ ) {
          this.blocks[i].audit( allowPendingBlock, allowPendingList, allowDirty );
        }
      }
    }
  } );
  
  BackboneDrawable.createDivBackbone = function() {
    var div = document.createElement( 'div' );
    div.style.position = 'absolute';
    div.style.left = '0';
    div.style.top = '0';
    div.style.width = '0';
    div.style.height = '0';
    return div;
  };
  
  /* jshint -W064 */
  Poolable( BackboneDrawable, {
    constructorDuplicateFactory: function( pool ) {
      return function( display, backboneInstance, transformRootInstance, renderer, isDisplayRoot ) {
        if ( pool.length ) {
          sceneryLog && sceneryLog.BackboneDrawable && sceneryLog.BackboneDrawable( 'new from pool' );
          return pool.pop().initialize( display, backboneInstance, transformRootInstance, renderer, isDisplayRoot );
        } else {
          sceneryLog && sceneryLog.BackboneDrawable && sceneryLog.BackboneDrawable( 'new from constructor' );
          return new BackboneDrawable( display, backboneInstance, transformRootInstance, renderer, isDisplayRoot );
        }
      };
    }
  } );
  
  return BackboneDrawable;
} );
