// Copyright 2002-2014, University of Colorado Boulder

/**
 * Renders a visual layer of WebGL drawables.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Sharfudeen Ashraf (For Ghent University)
 */
define( function( require ) {
  'use strict';

  // modules
  var scenery = require( 'SCENERY/scenery' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var FittedBlock = require( 'SCENERY/display/FittedBlock' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var Util = require( 'SCENERY/util/Util' );

  scenery.WebGLBlock = function WebGLBlock( display, renderer, transformRootInstance, filterRootInstance ) {
    this.initialize( display, renderer, transformRootInstance, filterRootInstance );
  };
  var WebGLBlock = scenery.WebGLBlock;

  inherit( FittedBlock, WebGLBlock, {
    initialize: function( display, renderer, transformRootInstance, filterRootInstance ) {

      this.initializeFittedBlock( display, renderer, transformRootInstance );

      // WebGLBlocks are hard-coded to take the full display size (as opposed to svg and canvas)
      // Since we saw some jitter on iPad, see #318 and generally expect WebGL layers to span the entire display
      // In the future, it would be good to understand what was causing the problem and make webgl consistent
      // with svg and canvas again.
      this.setFit( FittedBlock.FULL_DISPLAY );

      this.filterRootInstance = filterRootInstance;

      // TODO: This block can be shared across displays, so we need to handle preserveDrawingBuffer separately?
      this.preserveDrawingBuffer = display.options.preserveDrawingBuffer;

      this.dirtyDrawables = cleanArray( this.dirtyDrawables );

      if ( !this.domElement ) {
        this.canvas = document.createElement( 'canvas' );
        this.canvas.style.position = 'absolute';
        this.canvas.style.left = '0';
        this.canvas.style.top = '0';
        this.canvas.style.pointerEvents = 'none';
        this.canvasId = this.canvas.id = 'scenery-webgl' + this.id;

        var contextOptions = {
          antialias: true,
          preserveDrawingBuffer: this.preserveDrawingBuffer // true: need to clear buffer and is slower
        };

        // we've already committed to using a WebGLBlock, so no use in a try-catch around our context attempt
        this.gl = this.canvas.getContext( 'webgl', contextOptions ) || this.canvas.getContext( 'experimental-webgl', contextOptions );
        assert && assert( this.gl, 'We should have a context by now' );
        var gl = this.gl;

        this.backingScale = Util.backingScale( gl );

        gl.clearColor( 0, 0, 0, 0 );
        gl.clear( gl.COLOR_BUFFER_BIT );
        gl.blendFunc( gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA );
        gl.enable( gl.BLEND );

        this.domElement = this.canvas;
      }

      // reset any fit transforms that were applied
      Util.prepareForTransform( this.canvas, false );
      Util.unsetTransform( this.canvas ); // clear out any transforms that could have been previously applied

      this.projectionMatrix = this.projectionMatrix || new Matrix3().setTo32Bit();
      // a column-major 3x3 array specifying our projection matrix for 2D points (homogenized to (x,y,1))
      this.projectionMatrixArray = this.projectionMatrix.entries;

      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( 'initialized #' + this.id );

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
      throw new Error( 'setSizeFitBounds unimplemented for WebGLBlock' );
    },

    update: function() {
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( 'update #' + this.id );
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.push();

      var gl = this.gl;

      if ( this.dirty && !this.disposed ) {
        this.dirty = false;

        while ( this.dirtyDrawables.length ) {
          this.dirtyDrawables.pop().update();
        }

        // udpate the fit BEFORE drawing, since it may change our offset
        this.updateFit();

        // finalX = 2 * x / display.width - 1
        // finalY = 1 - 2 * y / display.height
        // result = matrix * ( x, y, 1 )
        this.projectionMatrix.rowMajor( 2 / this.display.width, 0, -1,
                                        0, -2 / this.display.height, 1,
                                        0, 0, 1 );

        // if we created the context with preserveDrawingBuffer, we need to clear before rendering
        if ( this.preserveDrawingBuffer ) {
          gl.clear( gl.COLOR_BUFFER_BIT );
        }

        gl.viewport( 0.0, 0.0, this.canvas.width, this.canvas.height );

        //OHTWO TODO: PERFORMANCE: create an array for faster drawable iteration (this is probably a hellish memory access pattern)
        for ( var drawable = this.firstDrawable; drawable !== null; drawable = drawable.nextDrawable ) {
          // NOTE: only handles custom webgl drawables
          if ( drawable.webglRenderer === Renderer.webglCustom ) {
            drawable.draw();
          }

          // exit loop end case
          if ( drawable === this.lastDrawable ) { break; }
        }

        gl.flush();
      }

      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.pop();
    },

    dispose: function() {
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( 'dispose #' + this.id );

      // TODO: many things to dispose!?

      // clear references
      cleanArray( this.dirtyDrawables );

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

      drawable.initializeContext( this );
    },

    removeDrawable: function( drawable ) {
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( '#' + this.id + '.removeDrawable ' + drawable.toString() );

      FittedBlock.prototype.removeDrawable.call( this, drawable );
    },

    onIntervalChange: function( firstDrawable, lastDrawable ) {
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( '#' + this.id + '.onIntervalChange ' + firstDrawable.toString() + ' to ' + lastDrawable.toString() );

      FittedBlock.prototype.onIntervalChange.call( this, firstDrawable, lastDrawable );

      this.markDirty();
    },

    onPotentiallyMovedDrawable: function( drawable ) {
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( '#' + this.id + '.onPotentiallyMovedDrawable ' + drawable.toString() );
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.push();

      assert && assert( drawable.parentDrawable === this );

      this.markDirty();

      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.pop();
    },

    toString: function() {
      return 'WebGLBlock#' + this.id + '-' + FittedBlock.fitString[ this.fit ];
    }
  } );

  Poolable.mixin( WebGLBlock, {
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
