// Copyright 2013-2016, University of Colorado Boulder

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
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var Emitter = require( 'AXON/Emitter' );
  var FittedBlock = require( 'SCENERY/display/FittedBlock' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var scenery = require( 'SCENERY/scenery' );
  var ShaderProgram = require( 'SCENERY/util/ShaderProgram' );
  var SpriteSheet = require( 'SCENERY/util/SpriteSheet' );
  var Util = require( 'SCENERY/util/Util' );

  /**
   * @constructor
   * @extends FittedBlock
   * @mixes Poolable
   *
   * @param display
   * @param renderer
   * @param transformRootInstance
   * @param filterRootInstance
   */
  function WebGLBlock( display, renderer, transformRootInstance, filterRootInstance ) {
    this.initialize( display, renderer, transformRootInstance, filterRootInstance );
  }

  scenery.register( 'WebGLBlock', WebGLBlock );

  inherit( FittedBlock, WebGLBlock, {
    initialize: function( display, renderer, transformRootInstance, filterRootInstance ) {
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( 'initialize #' + this.id );
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.push();

      // WebGLBlocks are hard-coded to take the full display size (as opposed to svg and canvas)
      // Since we saw some jitter on iPad, see #318 and generally expect WebGL layers to span the entire display
      // In the future, it would be good to understand what was causing the problem and make webgl consistent
      // with svg and canvas again.
      this.initializeFittedBlock( display, renderer, transformRootInstance, FittedBlock.FULL_DISPLAY );

      // TODO: Uhh, is this not used?
      this.filterRootInstance = filterRootInstance;

      // {boolean} - Whether we pass this flag to the WebGL Context. It will store the contents displayed on the screen,
      // so that canvas.toDataURL() will work. It also requires clearing the context manually ever frame. Both incur
      // performance costs, so it should be false by default.
      // TODO: This block can be shared across displays, so we need to handle preserveDrawingBuffer separately?
      this.preserveDrawingBuffer = display.options.preserveDrawingBuffer;

      // list of {Drawable}s that need to be updated before we update
      this.dirtyDrawables = cleanArray( this.dirtyDrawables );

      // {Array.<SpriteSheet>}, permanent list of spritesheets for this block
      this.spriteSheets = this.spriteSheets || [];

      // Projection {Matrix3} that maps from Scenery's global coordinate frame to normalized device coordinates,
      // where x,y are both in the range [-1,1] from one side of the Canvas to the other.
      this.projectionMatrix = this.projectionMatrix || new Matrix3();

      // @private {Float32Array} - Column-major 3x3 array specifying our projection matrix for 2D points
      // (homogenized to (x,y,1))
      this.projectionMatrixArray = new Float32Array( 9 );

      // processor for custom WebGL drawables (e.g. WebGLNode)
      this.customProcessor = this.customProcessor || new WebGLBlock.CustomProcessor();

      // processor for drawing vertex-colored triangles (e.g. Path types)
      this.vertexColorPolygonsProcessor = this.vertexColorPolygonsProcessor || new WebGLBlock.VertexColorPolygons( this.projectionMatrixArray );

      // processor for drawing textured triangles (e.g. Image)
      this.texturedTrianglesProcessor = this.texturedTrianglesProcessor || new WebGLBlock.TexturedTrianglesProcessor( this.projectionMatrixArray );

      // @public {Emitter} - Called when the WebGL context changes to a new context.
      this.glChangedEmitter = new Emitter();

      // @private {boolean}
      this.isContextLost = false;

      // @private {function}
      this.contextLostListener = this.onContextLoss.bind( this );
      this.contextRestoreListener = this.onContextRestoration.bind( this );

      if ( !this.domElement ) {
        // @public (scenery-internal) {HTMLCanvasElement} - Div wrapper used so we can switch out Canvases if necessary.
        this.domElement = document.createElement( 'div' );
        this.domElement.className = 'webgl-container';
        this.domElement.style.position = 'absolute';
        this.domElement.style.left = '0';
        this.domElement.style.top = '0';

        this.rebuildCanvas();
      }

      // clear buffers when we are reinitialized
      this.gl.clear( this.gl.COLOR_BUFFER_BIT );

      // reset any fit transforms that were applied
      Util.prepareForTransform( this.canvas, false ); // Apply CSS needed for future CSS transforms to work properly.
      Util.unsetTransform( this.canvas ); // clear out any transforms that could have been previously applied

      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.pop();

      return this;
    },

    /**
     * Forces a rebuild of the Canvas and its context (as long as a context can be obtained).
     * @private
     *
     * This can be necessary when the browser won't restore our context that was lost (and we need to create another
     * canvas to get a valid context).
     */
    rebuildCanvas: function() {
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( 'rebuildCanvas #' + this.id );
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.push();

      var canvas = document.createElement( 'canvas' );
      var gl = this.getContextFromCanvas( canvas );

      // Don't assert-failure out if this is not our first attempt (we're testing to see if we can recreate)
      assert && assert( gl || this.canvas, 'We should have a WebGL context by now' );

      // If we're aggressively trying to rebuild, we need to ignore context creation failure.
      if ( gl ) {
        if ( this.canvas ) {
          this.domElement.removeChild( this.canvas );
          this.canvas.removeEventListener( 'webglcontextlost', this.contextLostListener, false );
          this.canvas.removeEventListener( 'webglcontextrestored', this.contextRestoreListener, false );
        }

        // @private {HTMLCanvasElement}
        this.canvas = canvas;
        this.canvas.style.pointerEvents = 'none';

        // @private {number} - unique ID so that we can support rasterization with Display.foreignObjectRasterization
        this.canvasId = this.canvas.id = 'scenery-webgl' + this.id;

        this.canvas.addEventListener( 'webglcontextlost', this.contextLostListener, false );
        this.canvas.addEventListener( 'webglcontextrestored', this.contextRestoreListener, false );

        this.domElement.appendChild( this.canvas );

        this.setupContext( gl );
      }

      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.pop();
    },

    /**
     * Takes a fresh WebGL context switches the WebGL block over to use it.
     * @private
     *
     * @param {WebGLRenderingContext} gl
     */
    setupContext: function( gl ) {
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( 'setupContext #' + this.id );
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.push();

      assert && assert( gl, 'Should have an actual context if this is called' );

      this.isContextLost = false;

      // @private {WebGLRenderingContext}
      this.gl = gl;

      // @private {number} - How much larger our Canvas will be compared to the CSS pixel dimensions, so that our
      // Canvas maps one of its pixels to a physical pixel (for Retina devices, etc.).
      this.backingScale = Util.backingScale( this.gl );

      // Double the backing scale size if we detect no built-in antialiasing.
      // See https://github.com/phetsims/circuit-construction-kit-dc/issues/139 and
      // https://github.com/phetsims/scenery/issues/859.
      if ( this.display._allowBackingScaleAntialiasing && gl.getParameter( gl.SAMPLES ) === 0 ) {
        this.backingScale *= 2;
      }

      // @private {number}
      this.originalBackingScale = this.backingScale;

      Util.applyWebGLContextDefaults( this.gl ); // blending defaults, etc.

      // When the context changes, we need to force certain refreshes
      this.markDirty();
      this.dirtyFit = true; // Force re-fitting

      // Update the context references on the processors
      this.customProcessor.initializeContext( this.gl );
      this.vertexColorPolygonsProcessor.initializeContext( this.gl );
      this.texturedTrianglesProcessor.initializeContext( this.gl );

      // Notify spritesheets of the new context
      for ( var i = 0; i < this.spriteSheets.length; i++ ) {
        this.spriteSheets[ i ].initializeContext( this.gl );
      }

      // Notify (e.g. WebGLNode painters need to be recreated)
      this.glChangedEmitter.emit();

      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.pop();
    },

    /**
     * Attempts to force a Canvas rebuild to get a new Canvas/context pair.
     * @private
     */
    delayedRebuildCanvas: function() {
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( 'Delaying rebuilding of Canvas #' + this.id );
      var self = this;

      // TODO: Can we move this to before the update() step? Could happen same-frame in that case.
      window.setTimeout( function() {
        sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( 'Executing delayed rebuilding #' + this.id );
        sceneryLog && sceneryLog.WebGLBlock && sceneryLog.push();
        self.rebuildCanvas();
        sceneryLog && sceneryLog.WebGLBlock && sceneryLog.pop();
      } );
    },

    /**
     * Callback for whenever our WebGL context is lost.
     * @private
     *
     * @param {WebGLContextEvent} domEvent
     */
    onContextLoss: function( domEvent ) {
      if ( !this.isContextLost ) {
        sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( 'Context lost #' + this.id );
        sceneryLog && sceneryLog.WebGLBlock && sceneryLog.push();

        this.isContextLost = true;

        // Preventing default is super-important, otherwise it never attempts to restore the context
        domEvent.preventDefault();

        this.canvas.style.display = 'none';

        this.markDirty();

        sceneryLog && sceneryLog.WebGLBlock && sceneryLog.pop();
      }
    },

    /**
     * Callback for whenever our WebGL context is restored.
     * @private
     *
     * @param {WebGLContextEvent} domEvent
     */
    onContextRestoration: function( domEvent ) {
      if ( this.isContextLost ) {
        sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( 'Context restored #' + this.id );
        sceneryLog && sceneryLog.WebGLBlock && sceneryLog.push();

        var gl = this.getContextFromCanvas( this.canvas );
        assert && assert( gl, 'We were told the context was restored, so this should work' );

        this.setupContext( gl );

        this.canvas.style.display = '';

        sceneryLog && sceneryLog.WebGLBlock && sceneryLog.pop();
      }
    },

    /**
     * Attempts to get a WebGL context from a Canvas.
     * @private
     *
     * @param {HTMLCanvasElement}
     * @returns {WebGLRenderingContext|*} - If falsy, it did not succeed.
     */
    getContextFromCanvas: function( canvas ) {
      var contextOptions = {
        antialias: true,
        preserveDrawingBuffer: this.preserveDrawingBuffer
        // NOTE: we use premultiplied alpha since it should have better performance AND it appears to be the only one
        // truly compatible with texture filtering/interpolation.
        // See https://github.com/phetsims/energy-skate-park/issues/39, https://github.com/phetsims/scenery/issues/397
        // and https://stackoverflow.com/questions/39341564/webgl-how-to-correctly-blend-alpha-channel-png
      };

      // we've already committed to using a WebGLBlock, so no use in a try-catch around our context attempt
      return canvas.getContext( 'webgl', contextOptions ) || canvas.getContext( 'experimental-webgl', contextOptions );
    },

    setSizeFullDisplay: function() {
      var size = this.display.getSize();
      this.canvas.width = Math.ceil( size.width * this.backingScale );
      this.canvas.height = Math.ceil( size.height * this.backingScale );
      this.canvas.style.width = size.width + 'px';
      this.canvas.style.height = size.height + 'px';
    },

    setSizeFitBounds: function() {
      throw new Error( 'setSizeFitBounds unimplemented for WebGLBlock' );
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
      if ( !FittedBlock.prototype.update.call( this ) ) {
        return false;
      }

      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( 'update #' + this.id );
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.push();

      var gl = this.gl;

      if ( this.isContextLost && this.display._aggressiveContextRecreation ) {
        this.delayedRebuildCanvas();
      }

      // update drawables, so that they have vertex arrays up to date, etc.
      while ( this.dirtyDrawables.length ) {
        this.dirtyDrawables.pop().update();
      }

      // ensure sprite sheet textures are up-to-date
      var numSpriteSheets = this.spriteSheets.length;
      for ( var i = 0; i < numSpriteSheets; i++ ) {
        this.spriteSheets[ i ].updateTexture();
      }

      // temporary hack for supporting webglScale
      if ( this.firstDrawable &&
           this.firstDrawable === this.lastDrawable &&
           this.firstDrawable.node &&
           this.firstDrawable.node._hints.webglScale !== null &&
           this.backingScale !== this.originalBackingScale * this.firstDrawable.node._hints.webglScale ) {
        this.backingScale = this.originalBackingScale * this.firstDrawable.node._hints.webglScale;
        this.dirtyFit = true;
      }

      // udpate the fit BEFORE drawing, since it may change our offset
      this.updateFit();

      // finalX = 2 * x / display.width - 1
      // finalY = 1 - 2 * y / display.height
      // result = matrix * ( x, y, 1 )
      this.projectionMatrix.rowMajor(
        2 / this.display.width, 0, -1,
        0, -2 / this.display.height, 1,
        0, 0, 1 );
      this.projectionMatrix.copyToArray( this.projectionMatrixArray );

      // if we created the context with preserveDrawingBuffer, we need to clear before rendering
      if ( this.preserveDrawingBuffer ) {
        gl.clear( gl.COLOR_BUFFER_BIT );
      }

      gl.viewport( 0.0, 0.0, this.canvas.width, this.canvas.height );

      // We switch between processors for drawables based on each drawable's webglRenderer property. Each processor
      // will be activated, will process a certain number of adjacent drawables with that processor's webglRenderer,
      // and then will be deactivated. This allows us to switch back-and-forth between different shader programs,
      // and allows us to trigger draw calls for each grouping of drawables in an efficient way.
      var currentProcessor = null;
      // How many draw calls have been executed. If no draw calls are executed while updating, it means nothing should
      // be drawn, and we'll have to manually clear the Canvas if we are not preserving the drawing buffer.
      var cumulativeDrawCount = 0;
      // Iterate through all of our drawables (linked list)
      //OHTWO TODO: PERFORMANCE: create an array for faster drawable iteration (this is probably a hellish memory access pattern)
      for ( var drawable = this.firstDrawable; drawable !== null; drawable = drawable.nextDrawable ) {
        // ignore invisible drawables
        if ( drawable.visible ) {
          // select our desired processor
          var desiredProcessor = null;
          if ( drawable.webglRenderer === Renderer.webglTexturedTriangles ) {
            desiredProcessor = this.texturedTrianglesProcessor;
          }
          else if ( drawable.webglRenderer === Renderer.webglCustom ) {
            desiredProcessor = this.customProcessor;
          }
          else if ( drawable.webglRenderer === Renderer.webglVertexColorPolygons ) {
            desiredProcessor = this.vertexColorPolygonsProcessor;
          }
          assert && assert( desiredProcessor );

          // swap processors if necessary
          if ( desiredProcessor !== currentProcessor ) {
            // deactivate any old processors
            if ( currentProcessor ) {
              cumulativeDrawCount += currentProcessor.deactivate();
            }
            // activate the new processor
            currentProcessor = desiredProcessor;
            currentProcessor.activate();
          }

          // process our current drawable with the current processor
          currentProcessor.processDrawable( drawable );
        }

        // exit loop end case
        if ( drawable === this.lastDrawable ) { break; }
      }
      // deactivate any processor that still has drawables that need to be handled
      if ( currentProcessor ) {
        cumulativeDrawCount += currentProcessor.deactivate();
      }

      // If we executed no draw calls AND we aren't preserving the drawing buffer, we'll need to manually clear the
      // drawing buffer ourself.
      if ( cumulativeDrawCount === 0 && !this.preserveDrawingBuffer ) {
        gl.clear( gl.COLOR_BUFFER_BIT );
      }

      gl.flush();

      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.pop();

      return true;
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
      assert && assert( !drawable.isDisposed );

      // TODO: instance check to see if it is a canvas cache (usually we don't need to call update on our drawables)
      this.dirtyDrawables.push( drawable );
      this.markDirty();
    },

    addDrawable: function( drawable ) {
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( '#' + this.id + '.addDrawable ' + drawable.toString() );

      FittedBlock.prototype.addDrawable.call( this, drawable );

      // will trigger changes to the spritesheets for images, or initialization for others
      drawable.onAddToBlock( this );
    },

    removeDrawable: function( drawable ) {
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( '#' + this.id + '.removeDrawable ' + drawable.toString() );

      // Ensure a removed drawable is not present in the dirtyDrawables array afterwards. Don't want to update it.
      // See https://github.com/phetsims/scenery/issues/635
      var index = 0;
      while ( ( index = this.dirtyDrawables.indexOf( drawable, index ) ) >= 0 ) {
        this.dirtyDrawables.splice( index, 1 );
      }

      // wil trigger removal from spritesheets
      drawable.onRemoveFromBlock( this );

      FittedBlock.prototype.removeDrawable.call( this, drawable );
    },

    /**
     * Ensures we have an allocated part of a SpriteSheet for this image. If a SpriteSheet already contains this image,
     * we'll just increase the reference count. Otherwise, we'll attempt to add it into one of our SpriteSheets. If
     * it doesn't fit, we'll add a new SpriteSheet and add the image to it.
     *
     * @param {HTMLImageElement | HTMLCanvasElement} image
     * @param {number} width
     * @param {number} height
     *
     * @returns {Sprite} - Throws an error if we can't accommodate the image
     */
    addSpriteSheetImage: function( image, width, height ) {
      var sprite = null;
      var numSpriteSheets = this.spriteSheets.length;
      // TODO: check for SpriteSheet containment first?
      for ( var i = 0; i < numSpriteSheets; i++ ) {
        var spriteSheet = this.spriteSheets[ i ];
        sprite = spriteSheet.addImage( image, width, height );
        if ( sprite ) {
          break;
        }
      }
      if ( !sprite ) {
        var newSpriteSheet = new SpriteSheet( true ); // use mipmaps for now?
        sprite = newSpriteSheet.addImage( image, width, height );
        newSpriteSheet.initializeContext( this.gl );
        this.spriteSheets.push( newSpriteSheet );
        if ( !sprite ) {
          // TODO: renderer flags should change for very large images
          throw new Error( 'Attempt to load image that is too large for sprite sheets' );
        }
      }
      return sprite;
    },

    /**
     * Removes the reference to the sprite in our spritesheets.
     *
     * @param {Sprite} sprite
     */
    removeSpriteSheetImage: function( sprite ) {
      sprite.spriteSheet.removeImage( sprite.image );
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

  /*---------------------------------------------------------------------------*
   * Processors rely on the following lifecycle:
   * 1. activate()
   * 2. processDrawable() - 0 or more times
   * 3. deactivate()
   * Once deactivated, they should have executed all of the draw calls they need to make.
   *---------------------------------------------------------------------------*/

  // TODO: Processor super-type?

  /**
   * @constructor
   */
  WebGLBlock.CustomProcessor = function() {
    this.drawable = null;
  };
  inherit( Object, WebGLBlock.CustomProcessor, {
    /**
     * Sets the WebGL context that this processor should use.
     * @public
     *
     * NOTE: This can be called multiple times on a single processor, in the case where the previous context was lost.
     *       We should not need to dispose anything from that.
     *
     * @param {WebGLRenderingContext} gl
     */
    initializeContext: function( gl ) {

    },

    activate: function() {
      this.drawCount = 0;
    },

    processDrawable: function( drawable ) {
      assert && assert( drawable.webglRenderer === Renderer.webglCustom );

      this.drawable = drawable;
      this.draw();
    },

    deactivate: function() {
      return this.drawCount;
    },

    // @private
    draw: function() {
      if ( this.drawable ) {
        var count = this.drawable.draw();
        assert && assert( typeof count === 'number' );
        this.drawCount += count;
        this.drawable = null;
      }
    }
  } );

  /**
   * @constructor
   *
   * @param {Float32Array} projectionMatrixArray - Projection matrix entries
   */
  WebGLBlock.VertexColorPolygons = function( projectionMatrixArray ) {
    assert && assert( projectionMatrixArray instanceof Float32Array );

    // @private {Float32Array}
    this.projectionMatrixArray = projectionMatrixArray;

    // @private {number} - Initial length of the vertex buffer. May increase as needed.
    this.lastArrayLength = 128;

    // @private {Float32Array}
    this.vertexArray = new Float32Array( this.lastArrayLength );
  };
  inherit( Object, WebGLBlock.VertexColorPolygons, {
    /**
     * Sets the WebGL context that this processor should use.
     * @public
     *
     * NOTE: This can be called multiple times on a single processor, in the case where the previous context was lost.
     *       We should not need to dispose anything from that.
     *
     * @param {WebGLRenderingContext} gl
     */
    initializeContext: function( gl ) {
      assert && assert( gl, 'Should be an actual context' );

      // @private {WebGLRenderingContext}
      this.gl = gl;

      // @private {ShaderProgram}
      this.shaderProgram = new ShaderProgram( gl, [
        // vertex shader
        'attribute vec2 aVertex;',
        'attribute vec4 aColor;',
        'varying vec4 vColor;',
        'uniform mat3 uProjectionMatrix;',

        'void main() {',
        '  vColor = aColor;',
        '  vec3 ndc = uProjectionMatrix * vec3( aVertex, 1.0 );', // homogeneous map to to normalized device coordinates
        '  gl_Position = vec4( ndc.xy, 0.0, 1.0 );',
        '}'
      ].join( '\n' ), [
        // fragment shader
        'precision mediump float;',
        'varying vec4 vColor;',

        'void main() {',
        // NOTE: Premultiplying alpha here is needed since we're going back to the standard blend functions.
        // See https://github.com/phetsims/energy-skate-park/issues/39, https://github.com/phetsims/scenery/issues/397
        // and https://stackoverflow.com/questions/39341564/webgl-how-to-correctly-blend-alpha-channel-png
        '  gl_FragColor = vec4( vColor.rgb * vColor.a, vColor.a );',
        '}'
      ].join( '\n' ), {
        attributes: [ 'aVertex', 'aColor' ],
        uniforms: [ 'uProjectionMatrix' ]
      } );

      // @private {WebGLBuffer}
      this.vertexBuffer = gl.createBuffer();

      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
      gl.bufferData( gl.ARRAY_BUFFER, this.vertexArray, gl.DYNAMIC_DRAW ); // fully buffer at the start
    },

    activate: function() {
      this.shaderProgram.use();

      this.vertexArrayIndex = 0;
      this.drawCount = 0;
    },

    processDrawable: function( drawable ) {
      if ( drawable.includeVertices ) {
        var vertexData = drawable.vertexArray;

        // if our vertex data won't fit, keep doubling the size until it fits
        while ( vertexData.length + this.vertexArrayIndex > this.vertexArray.length ) {
          var newVertexArray = new Float32Array( this.vertexArray.length * 2 );
          newVertexArray.set( this.vertexArray );
          this.vertexArray = newVertexArray;
        }

        // copy our vertex data into the main array
        this.vertexArray.set( vertexData, this.vertexArrayIndex );
        this.vertexArrayIndex += vertexData.length;

        this.drawCount++;
      }
    },

    deactivate: function() {
      if ( this.drawCount ) {
        this.draw();
      }

      this.shaderProgram.unuse();

      return this.drawCount;
    },

    // @private
    draw: function() {
      var gl = this.gl;

      // (uniform) projection transform into normalized device coordinates
      gl.uniformMatrix3fv( this.shaderProgram.uniformLocations.uProjectionMatrix, false, this.projectionMatrixArray );

      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
      // if we increased in length, we need to do a full bufferData to resize it on the GPU side
      if ( this.vertexArray.length > this.lastArrayLength ) {
        gl.bufferData( gl.ARRAY_BUFFER, this.vertexArray, gl.DYNAMIC_DRAW ); // fully buffer at the start
      }
      // otherwise do a more efficient update that only sends part of the array over
      else {
        gl.bufferSubData( gl.ARRAY_BUFFER, 0, this.vertexArray.subarray( 0, this.vertexArrayIndex ) );
      }
      var sizeOfFloat = Float32Array.BYTES_PER_ELEMENT;
      var stride = 6 * sizeOfFloat;
      gl.vertexAttribPointer( this.shaderProgram.attributeLocations.aVertex, 2, gl.FLOAT, false, stride, 0 * sizeOfFloat );
      gl.vertexAttribPointer( this.shaderProgram.attributeLocations.aColor, 4, gl.FLOAT, false, stride, 2 * sizeOfFloat );

      gl.drawArrays( gl.TRIANGLES, 0, this.vertexArrayIndex / 6 );

      this.vertexArrayIndex = 0;
    }
  } );

  /**
   * @constructor
   *
   * @param {Float32Array} projectionMatrixArray - Projection matrix entries
   */
  WebGLBlock.TexturedTrianglesProcessor = function( projectionMatrixArray ) {
    assert && assert( projectionMatrixArray instanceof Float32Array );

    // @private {Float32Array}
    this.projectionMatrixArray = projectionMatrixArray;

    // @private {number} - Initial length of the vertex buffer. May increase as needed.
    this.lastArrayLength = 128;

    // @private {Float32Array}
    this.vertexArray = new Float32Array( this.lastArrayLength );
  };
  inherit( Object, WebGLBlock.TexturedTrianglesProcessor, {
    /**
     * Sets the WebGL context that this processor should use.
     * @public
     *
     * NOTE: This can be called multiple times on a single processor, in the case where the previous context was lost.
     *       We should not need to dispose anything from that.
     *
     * @param {WebGLRenderingContext} gl
     */
    initializeContext: function( gl ) {
      assert && assert( gl, 'Should be an actual context' );

      // @private {WebGLRenderingContext}
      this.gl = gl;

      // @private {ShaderProgram}
      this.shaderProgram = new ShaderProgram( gl, [
        // vertex shader
        'attribute vec2 aVertex;',
        'attribute vec2 aTextureCoord;',
        'attribute float aAlpha;',
        'varying vec2 vTextureCoord;',
        'varying float vAlpha;',
        'uniform mat3 uProjectionMatrix;',

        'void main() {',
        '  vTextureCoord = aTextureCoord;',
        '  vAlpha = aAlpha;',
        '  vec3 ndc = uProjectionMatrix * vec3( aVertex, 1.0 );', // homogeneous map to to normalized device coordinates
        '  gl_Position = vec4( ndc.xy, 0.0, 1.0 );',
        '}'
      ].join( '\n' ), [
        // fragment shader
        'precision mediump float;',
        'varying vec2 vTextureCoord;',
        'varying float vAlpha;',
        'uniform sampler2D uTexture;',

        'void main() {',
        '  vec4 color = texture2D( uTexture, vTextureCoord, -0.7 );', // mipmap LOD bias of -0.7 (for now)
        '  color.a *= vAlpha;',
        '  gl_FragColor = color;', // don't premultiply alpha (we are loading the textures as premultiplied already)
        '}'
      ].join( '\n' ), {
        // attributes: [ 'aVertex', 'aTextureCoord' ],
        attributes: [ 'aVertex', 'aTextureCoord', 'aAlpha' ],
        uniforms: [ 'uTexture', 'uProjectionMatrix' ]
      } );

      // @private {WebGLBuffer}
      this.vertexBuffer = gl.createBuffer();

      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
      gl.bufferData( gl.ARRAY_BUFFER, this.vertexArray, gl.DYNAMIC_DRAW ); // fully buffer at the start
    },

    activate: function() {
      this.shaderProgram.use();

      this.currentSpriteSheet = null;
      this.vertexArrayIndex = 0;
      this.drawCount = 0;
    },

    processDrawable: function( drawable ) {
      // skip unloaded images or sprites
      if ( !drawable.sprite ) {
        return;
      }

      assert && assert( drawable.webglRenderer === Renderer.webglTexturedTriangles );
      if ( this.currentSpriteSheet && drawable.sprite.spriteSheet !== this.currentSpriteSheet ) {
        this.draw();
      }
      this.currentSpriteSheet = drawable.sprite.spriteSheet;

      var vertexData = drawable.vertexArray;

      // if our vertex data won't fit, keep doubling the size until it fits
      while ( vertexData.length + this.vertexArrayIndex > this.vertexArray.length ) {
        var newVertexArray = new Float32Array( this.vertexArray.length * 2 );
        newVertexArray.set( this.vertexArray );
        this.vertexArray = newVertexArray;
      }

      // copy our vertex data into the main array
      this.vertexArray.set( vertexData, this.vertexArrayIndex );
      this.vertexArrayIndex += vertexData.length;
    },

    deactivate: function() {
      if ( this.currentSpriteSheet ) {
        this.draw();
      }

      this.shaderProgram.unuse();

      return this.drawCount;
    },

    // @private
    draw: function() {
      assert && assert( this.currentSpriteSheet );
      var gl = this.gl;

      // (uniform) projection transform into normalized device coordinates
      gl.uniformMatrix3fv( this.shaderProgram.uniformLocations.uProjectionMatrix, false, this.projectionMatrixArray );

      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
      // if we increased in length, we need to do a full bufferData to resize it on the GPU side
      if ( this.vertexArray.length > this.lastArrayLength ) {
        gl.bufferData( gl.ARRAY_BUFFER, this.vertexArray, gl.DYNAMIC_DRAW ); // fully buffer at the start
      }
      // otherwise do a more efficient update that only sends part of the array over
      else {
        gl.bufferSubData( gl.ARRAY_BUFFER, 0, this.vertexArray.subarray( 0, this.vertexArrayIndex ) );
      }

      var numComponents = 5;
      var sizeOfFloat = Float32Array.BYTES_PER_ELEMENT;
      var stride = numComponents * sizeOfFloat;
      gl.vertexAttribPointer( this.shaderProgram.attributeLocations.aVertex, 2, gl.FLOAT, false, stride, 0 * sizeOfFloat );
      gl.vertexAttribPointer( this.shaderProgram.attributeLocations.aTextureCoord, 2, gl.FLOAT, false, stride, 2 * sizeOfFloat );
      gl.vertexAttribPointer( this.shaderProgram.attributeLocations.aAlpha, 1, gl.FLOAT, false, stride, 4 * sizeOfFloat );

      gl.activeTexture( gl.TEXTURE0 );
      gl.bindTexture( gl.TEXTURE_2D, this.currentSpriteSheet.texture );
      gl.uniform1i( this.shaderProgram.uniformLocations.uTexture, 0 );

      gl.drawArrays( gl.TRIANGLES, 0, this.vertexArrayIndex / numComponents );

      gl.bindTexture( gl.TEXTURE_2D, null );

      this.drawCount++;

      this.currentSpriteSheet = null;
      this.vertexArrayIndex = 0;
    }
  } );

  Poolable.mixInto( WebGLBlock, {
    initialize: WebGLBlock.prototype.initialize
  } );

  return WebGLBlock;
} );
