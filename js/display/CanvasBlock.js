// Copyright 2002-2014, University of Colorado

/**
 * Handles a visual Canvas layer of drawables.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var Vector2 = require( 'DOT/Vector2' );
  var scenery = require( 'SCENERY/scenery' );
  var FittedBlock = require( 'SCENERY/display/FittedBlock' );
  var CanvasContextWrapper = require( 'SCENERY/util/CanvasContextWrapper' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var Util = require( 'SCENERY/util/Util' );
  
  scenery.CanvasBlock = function CanvasBlock( display, renderer, transformRootInstance ) {
    this.initialize( display, renderer, transformRootInstance );
  };
  var CanvasBlock = scenery.CanvasBlock;
  
  inherit( FittedBlock, CanvasBlock, {
    initialize: function( display, renderer, transformRootInstance ) {
      this.initializeFittedBlock( display, renderer, transformRootInstance );
      
      this.dirtyDrawables = cleanArray( this.dirtyDrawables );
      
      if ( !this.domElement ) {
        //OHTWO TODO: support tiled Canvas handling (will need to wrap then in a div, or something)
        this.canvas = document.createElement( 'canvas' );
        this.canvas.style.position = 'absolute';
        this.canvas.style.left = '0';
        this.canvas.style.top = '0';
        
        this.context = this.canvas.getContext( '2d' );
        
        // workaround for Chrome (WebKit) miterLimit bug: https://bugs.webkit.org/show_bug.cgi?id=108763
        this.context.miterLimit = 20;
        this.context.miterLimit = 10;
        
        this.wrapper = new CanvasContextWrapper( this.canvas, this.context );
        
        this.domElement = this.canvas;
      }
      
      // reset any fit transforms that were applied
      Util.prepareForTransform( this.canvas, this.forceAcceleration );
      Util.unsetTransform( this.canvas ); // clear out any transforms that could have been previously applied
      
      this.canvasDrawOffset = new Vector2();
      
      // store our backing scale so we don't have to look it up while fitting
      this.backingScale = ( renderer & Renderer.bitmaskCanvasLowResolution ) ? 1 : scenery.Util.backingScale( this.context );
      
      sceneryLayerLog && sceneryLayerLog.CanvasBlock && sceneryLayerLog.CanvasBlock( 'initialized #' + this.id );
      // TODO: dirty list of nodes (each should go dirty only once, easier than scanning all?)
      
      return this;
    },
    
    setSizeFullDisplay: function() {
      var size = this.display.getSize();
      this.canvas.width = size.width * this.backingScale;
      this.canvas.height = size.height * this.backingScale;
      this.canvas.style.width = size.width + 'px';
      this.canvas.style.height = size.height + 'px';
    },
    
    setSizeFitBounds: function() {
      var x = this.fitBounds.minX;
      var y = this.fitBounds.minY;
      this.canvasDrawOffset.setXY( -x, -y ); // subtract off so we have a tight fit
      //OHTWO TODO PERFORMANCE: see if we can get a speedup by putting the backing scale in our transform instead of with CSS?
      Util.setTransform( 'matrix(1,0,0,1,' + x + ',' + y + ')', this.canvas, this.forceAcceleration ); // reapply the translation as a CSS transform
      this.canvas.width = this.fitBounds.width * this.backingScale;
      this.canvas.height = this.fitBounds.height * this.backingScale;
      this.canvas.style.width = this.fitBounds.width + 'px';
      this.canvas.style.height = this.fitBounds.height + 'px';
    },
    
    update: function() {
      sceneryLayerLog && sceneryLayerLog.CanvasBlock && sceneryLayerLog.CanvasBlock( 'update #' + this.id );
      
      if ( this.dirty && !this.disposed ) {
        this.dirty = false;
        
        while ( this.dirtyDrawables.length ) {
          this.dirtyDrawables.pop().update();
        }
        
        // udpate the fit BEFORE drawing, since it may change our offset
        this.updateFit();
        
        // for now, clear everything!
        this.context.setTransform( 1, 0, 0, 1, 0, 0 ); // identity
        this.context.clearRect( 0, 0, this.canvas.width, this.canvas.height ); // clear everything
        
        //OHTWO TODO: PERFORMANCE: create an array for faster drawable iteration (this is probably a hellish memory access pattern)
        for ( var drawable = this.firstDrawable; drawable !== null && drawable !== this.lastDrawable; drawable = drawable.nextDrawable ) {
          this.renderDrawable( drawable );
        }
        // iteration above skips last drawable
        this.lastDrawable && this.renderDrawable( drawable );
      }
    },
    
    renderDrawable: function( drawable ) {
      // we're directly accessing the relative transform below, so we need to ensure that it is up-to-date
      assert && assert( drawable.instance.isValidationNotNeeded() );
      
      var matrix = drawable.instance.relativeMatrix;
      
      // set the correct (relative to the transform root) transform up, instead of walking the hierarchy (for now)
      //OHTWO TODO: are the offsets specified in the correct order with the backingScale?
      this.context.setTransform( this.backingScale, 0, 0, this.backingScale, this.canvasDrawOffset.x * this.backingScale, this.canvasDrawOffset.y * this.backingScale );
      //OHTWO TODO: should we start premultiplying these matrices to remove this bottleneck?
      matrix.canvasAppendTransform( this.context );
      
      // paint using its local coordinate frame
      drawable.paintCanvas( this.wrapper, drawable.instance.node );
    },
    
    dispose: function() {
      sceneryLayerLog && sceneryLayerLog.CanvasBlock && sceneryLayerLog.CanvasBlock( 'dispose #' + this.id );
      
      // clear references
      this.transformRootInstance = null;
      cleanArray( this.dirtyDrawables );
      
      // minimize memory exposure of the backing raster
      this.canvas.width = 0;
      this.canvas.height = 0;
      
      FittedBlock.prototype.dispose.call( this );
    },
    
    markDirtyDrawable: function( drawable ) {
      sceneryLayerLog && sceneryLayerLog.dirty && sceneryLayerLog.dirty( 'markDirtyDrawable on CanvasBlock#' + this.id + ' with ' + drawable.toString() );
      
      assert && assert( drawable );
      
      // TODO: instance check to see if it is a canvas cache (usually we don't need to call update on our drawables)
      this.dirtyDrawables.push( drawable );
      this.markDirty();
    },
    
    addDrawable: function( drawable ) {
      sceneryLayerLog && sceneryLayerLog.CanvasBlock && sceneryLayerLog.CanvasBlock( '#' + this.id + '.addDrawable ' + drawable.toString() );
      
      FittedBlock.prototype.addDrawable.call( this, drawable );
    },
    
    removeDrawable: function( drawable ) {
      sceneryLayerLog && sceneryLayerLog.CanvasBlock && sceneryLayerLog.CanvasBlock( '#' + this.id + '.removeDrawable ' + drawable.toString() );
      
      FittedBlock.prototype.removeDrawable.call( this, drawable );
    },
    
    notifyInterval: function( firstDrawable, lastDrawable ) {
      sceneryLayerLog && sceneryLayerLog.CanvasBlock && sceneryLayerLog.CanvasBlock( '#' + this.id + '.notifyInterval ' + firstDrawable.toString() + ' to ' + lastDrawable.toString() );
      
      FittedBlock.prototype.notifyInterval.call( this, firstDrawable, lastDrawable );
    },
    
    toString: function() {
      return 'CanvasBlock#' + this.id + ' ' + FittedBlock.fitString[this.fit];
    }
  } );
  
  /* jshint -W064 */
  Poolable( CanvasBlock, {
    constructorDuplicateFactory: function( pool ) {
      return function( display, renderer, transformRootInstance ) {
        if ( pool.length ) {
          sceneryLayerLog && sceneryLayerLog.CanvasBlock && sceneryLayerLog.CanvasBlock( 'new from pool' );
          return pool.pop().initialize( display, renderer, transformRootInstance );
        } else {
          sceneryLayerLog && sceneryLayerLog.CanvasBlock && sceneryLayerLog.CanvasBlock( 'new from constructor' );
          return new CanvasBlock( display, renderer, transformRootInstance );
        }
      };
    }
  } );
  
  return CanvasBlock;
} );
