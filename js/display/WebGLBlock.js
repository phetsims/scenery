// Copyright 2002-2014, University of Colorado Boulder


/**
 * Handles a visual WebGL layer of drawables.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var Vector2 = require( 'DOT/Vector2' );
  var Matrix4 = require( 'DOT/Matrix4' );
  var scenery = require( 'SCENERY/scenery' );
  var FittedBlock = require( 'SCENERY/display/FittedBlock' );
  var WebGLContextWrapper = require( 'SCENERY/util/WebGLContextWrapper' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var Util = require( 'SCENERY/util/Util' );

  scenery.WebGLBlock = function WebGLBlock( display, renderer, transformRootInstance, filterRootInstance ) {
    this.initialize( display, renderer, transformRootInstance, filterRootInstance );
  };
  var WebGLBlock = scenery.WebGLBlock;

  inherit( FittedBlock, WebGLBlock, {
    initialize: function( display, renderer, transformRootInstance, filterRootInstance ) {
      this.initializeFittedBlock( display, renderer, transformRootInstance );

      this.filterRootInstance = filterRootInstance;

      this.dirtyDrawables = cleanArray( this.dirtyDrawables );

      if ( !this.domElement ) {
        //OHTWO TODO: support tiled WebGL handling (will need to wrap then in a div, or something)
        this.canvas = document.createElement( 'canvas' );
        this.canvas.style.position = 'absolute';
        this.canvas.style.left = '0';
        this.canvas.style.top = '0';
        this.canvas.style.pointerEvents = 'none';

        this.context = this.canvas.getContext( '2d' );

        // workaround for Chrome (WebKit) miterLimit bug: https://bugs.webkit.org/show_bug.cgi?id=108763
        this.context.miterLimit = 20;
        this.context.miterLimit = 10;

        this.wrapper = new WebGLContextWrapper( this.canvas, this.context );

        this.domElement = this.canvas;
      }

      // (0,height) => (0, -2) => ( 1, -1 )
      this.projectionMatrix = this.projectionMatrix || new Matrix4();


      // Keep track of whether the context is lost, so that we can avoid trying to render while the context is lost.
      this.webglContextIsLost = false;

      // If the scene was instructed to make a WebGL context that can simulate context loss, wrap it here, see #279
      if ( this.scene.webglMakeLostContextSimulatingCanvas ) {
        this.canvas = WebGLDebugUtils.makeLostContextSimulatingCanvas( this.canvas );
      }

      // Callback for context loss, see #279
      this.canvas.addEventListener( "webglcontextlost", function( event ) {
        console.log( 'context lost' );

        // khronos does not explain why we must prevent default in webgl context loss, but we must do so:
        // http://www.khronos.org/webgl/wiki/HandlingContextLost#Handling_Lost_Context_in_WebGL
        event.preventDefault();
        webglLayer.webglContextIsLost = true;
      }, false );

      // Only used when webglLayer.scene.webglSimulateIncrementalContextLoss is defined
      var numCallsToLoseContext = 1;

      // Callback for context restore, see #279
      this.canvas.addEventListener( "webglcontextrestored", function( event ) {
        console.log( 'context restored' );
        webglLayer.webglContextIsLost = false;

        // When context is restored, optionally simulate another context loss at an increased number of gl calls
        // This is because we must test for context loss between every pair of gl calls
        if ( webglLayer.scene.webglContextLossIncremental ) {
          console.log( 'simulating context loss in ', numCallsToLoseContext, 'gl calls.' );
          webglLayer.canvas.loseContextInNCalls( numCallsToLoseContext );
          numCallsToLoseContext++;
        }

        // Reinitialize the layer state
        webglLayer.initialize();

        // Reinitialize the webgl state for every instance's drawable
        var length = webglLayer.instances.length;
        for ( var i = 0; i < length; i++ ) {
          webglLayer.instances[i].data.drawable.initialize();
        }

        // Mark for repainting
        webglLayer.dirty = true;
      }, false );


      // reset any fit transforms that were applied
      Util.prepareForTransform( this.canvas, this.forceAcceleration );
      Util.unsetTransform( this.canvas ); // clear out any transforms that could have been previously applied

      this.canvasDrawOffset = new Vector2();

      // store our backing scale so we don't have to look it up while fitting
      this.backingScale = ( renderer & Renderer.bitmaskWebGLLowResolution ) ? 1 : scenery.Util.backingScale( this.context );

      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( 'initialized #' + this.id );
      // TODO: dirty list of nodes (each should go dirty only once, easier than scanning all?)

      return this;
    },

    setSizeFullDisplay: function() {
      var size = this.display.getSize();
      this.canvas.width = size.width * this.backingScale;
      this.canvas.height = size.height * this.backingScale;
      this.canvas.style.width = size.width + 'px';
      this.canvas.style.height = size.height + 'px';
      this.wrapper.resetStyles();
      this.updateProjectionMatrix( size.width, size.height );
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
      this.wrapper.resetStyles();
      this.updateProjectionMatrix( this.fitBounds.width, this.fitBounds.width );
    },

    updateProjectionMatrix: function( width, height ) {
      this.projectionMatrix.set( Matrix4.translation( -1, 1, 0 ).timesMatrix( Matrix4.scaling( 2 / width, -2 / height, 1 ) ) );
    },

    update: function() {
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( 'update #' + this.id );
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.push();

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

        //OHTWO TODO: clipping handling!
        if ( this.filterRootInstance.node._clipArea ) {
          this.context.save();

          this.temporaryRecursiveClip( this.filterRootInstance );
        }

        //OHTWO TODO: PERFORMANCE: create an array for faster drawable iteration (this is probably a hellish memory access pattern)
        for ( var drawable = this.firstDrawable; drawable !== null; drawable = drawable.nextDrawable ) {
          this.renderDrawable( drawable );
          if ( drawable === this.lastDrawable ) { break; }
        }

        if ( this.filterRootInstance.node._clipArea ) {
          this.context.restore();
        }
      }

      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.pop();
    },

    //OHTWO TODO: rework and do proper clipping support
    temporaryRecursiveClip: function( instance ) {
      if ( instance.parent ) {
        this.temporaryRecursiveClip( instance.parent );
      }
      if ( instance.node._clipArea ) {
        //OHTWO TODO: reduce duplication here
        this.context.setTransform( this.backingScale, 0, 0, this.backingScale, this.canvasDrawOffset.x * this.backingScale, this.canvasDrawOffset.y * this.backingScale );
        instance.relativeMatrix.canvasAppendTransform( this.context );

        // do the clipping
        this.context.beginPath();
        instance.node._clipArea.writeToContext( this.context );
        this.context.clip();
      }
    },

    renderDrawable: function( drawable ) {
      // we're directly accessing the relative transform below, so we need to ensure that it is up-to-date
      assert && assert( drawable.instance.isValidationNotNeeded() );

      var matrix = drawable.instance.relativeMatrix;

      // set the correct (relative to the transform root) transform up, instead of walking the hierarchy (for now)
      //OHTWO TODO: should we start premultiplying these matrices to remove this bottleneck?
      this.context.setTransform( this.backingScale, 0, 0, this.backingScale, this.canvasDrawOffset.x * this.backingScale, this.canvasDrawOffset.y * this.backingScale );
      if ( drawable.instance !== this.transformRootInstance ) {
        matrix.canvasAppendTransform( this.context );
      }

      // paint using its local coordinate frame
      drawable.paintWebGL( this.wrapper, drawable.instance.node );
    },

    dispose: function() {
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( 'dispose #' + this.id );

      // clear references
      this.transformRootInstance = null;
      cleanArray( this.dirtyDrawables );

      // minimize memory exposure of the backing raster
      this.canvas.width = 0;
      this.canvas.height = 0;

      FittedBlock.prototype.dispose.call( this );
    },

    markDirtyDrawable: function( drawable ) {
      sceneryLog && sceneryLog.dirty && sceneryLog.dirty( 'markDirtyDrawable on WebGLBlock#' + this.id + ' with ' + drawable.toString() );

      assert && assert( drawable );

      // TODO: instance check to see if it is a canvas cache (usually we don't need to call update on our drawables)
      this.dirtyDrawables.push( drawable );
      this.markDirty();
    },

    addDrawable: function( drawable ) {
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( '#' + this.id + '.addDrawable ' + drawable.toString() );

      FittedBlock.prototype.addDrawable.call( this, drawable );
    },

    removeDrawable: function( drawable ) {
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( '#' + this.id + '.removeDrawable ' + drawable.toString() );

      FittedBlock.prototype.removeDrawable.call( this, drawable );
    },

    onIntervalChange: function( firstDrawable, lastDrawable ) {
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( '#' + this.id + '.onIntervalChange ' + firstDrawable.toString() + ' to ' + lastDrawable.toString() );

      FittedBlock.prototype.onIntervalChange.call( this, firstDrawable, lastDrawable );
    },

    onPotentiallyMovedDrawable: function( drawable ) {
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( '#' + this.id + '.onPotentiallyMovedDrawable ' + drawable.toString() );
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.push();

      assert && assert( drawable.parentDrawable === this );

      // For now, mark it as dirty so that we redraw anything containing it. In the future, we could have more advanced
      // behavior that figures out the intersection-region for what was moved and what it was moved past, but that's
      // a harder problem.
      drawable.markDirty();

      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.pop();
    },

    toString: function() {
      return 'WebGLBlock#' + this.id + '-' + FittedBlock.fitString[this.fit];
    }
  } );

  /* jshint -W064 */
  Poolable( WebGLBlock, {
    constructorDuplicateFactory: function( pool ) {
      return function( display, renderer, transformRootInstance, filterRootInstance ) {
        if ( pool.length ) {
          sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( 'new from pool' );
          return pool.pop().initialize( display, renderer, transformRootInstance, filterRootInstance );
        }
        else {
          sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( 'new from constructor' );
          return new WebGLBlock( display, renderer, transformRootInstance, filterRootInstance );
        }
      };
    }
  } );

  return WebGLBlock;
} );
