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
  var SpriteSheet = require( 'SCENERY/util/SpriteSheet' );
  var ShaderProgram = require( 'SCENERY/util/ShaderProgram' );

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

      // {boolean} - Whether we pass this flag to the WebGL Context. It will store the contents displayed on the screen,
      // so that canvas.toDataURL() will work. It also requires clearing the context manually ever frame. Both incur
      // performance costs, so it should be false by default.
      // TODO: This block can be shared across displays, so we need to handle preserveDrawingBuffer separately?
      this.preserveDrawingBuffer = display.options.preserveDrawingBuffer;

      // list of {Drawable}s that need to be updated before we update
      this.dirtyDrawables = cleanArray( this.dirtyDrawables );

      // {Array.<SpriteSheet>}, permanent list of spritesheets for this block
      this.spriteSheets = this.spriteSheets || [];

      if ( !this.domElement ) {
        this.canvas = document.createElement( 'canvas' );
        this.canvas.style.position = 'absolute';
        this.canvas.style.left = '0';
        this.canvas.style.top = '0';
        this.canvas.style.pointerEvents = 'none';

        // unique ID so that we can support rasterization with Display.foreignObjectRasterization
        this.canvasId = this.canvas.id = 'scenery-webgl' + this.id;

        var contextOptions = {
          antialias: true,
          preserveDrawingBuffer: this.preserveDrawingBuffer
        };

        // we've already committed to using a WebGLBlock, so no use in a try-catch around our context attempt
        this.gl = this.canvas.getContext( 'webgl', contextOptions ) || this.canvas.getContext( 'experimental-webgl', contextOptions );
        assert && assert( this.gl, 'We should have a context by now' );
        var gl = this.gl;

        // {number} - How much larger our Canvas will be compared to the CSS pixel dimensions, so that our Canvas maps
        // one of its pixels to a physical pixel (for Retina devices, etc.).
        this.backingScale = this.originalBackingScale = Util.backingScale( gl );

        // What color gets set when we call gl.clear()
        gl.clearColor( 0, 0, 0, 0 );

        // Blending similar to http://localhost/phet/git/webgl-blendfunctions/blendfuncseparate.html
        gl.enable( gl.BLEND );
        gl.blendEquationSeparate( gl.FUNC_ADD, gl.FUNC_ADD );
        gl.blendFuncSeparate( gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA );

        this.domElement = this.canvas;

        // processor for custom WebGL drawables (e.g. WebGLNode)
        this.customProcessor = new WebGLBlock.CustomProcessor( this );

        // processor for drawing textured triangles (e.g. Image)
        this.texturedTrianglesProcessor = new WebGLBlock.TexturedTrianglesProcessor( this );
      }

      // clear buffers when we are reinitialized
      this.gl.clear( this.gl.COLOR_BUFFER_BIT );

      // reset any fit transforms that were applied
      Util.prepareForTransform( this.canvas, false );
      Util.unsetTransform( this.canvas ); // clear out any transforms that could have been previously applied

      // Projection {Matrix3} that maps from Scenery's global coordinate frame to normalized device coordinates,
      // where x,y are both in the range [-1,1] from one side of the Canvas to the other.
      this.projectionMatrix = this.projectionMatrix || new Matrix3().setTo32Bit();
      // a column-major 3x3 array specifying our projection matrix for 2D points (homogenized to (x,y,1))
      this.projectionMatrixArray = this.projectionMatrix.entries;

      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( 'initialized #' + this.id );

      return this;
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

    update: function() {
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( 'update #' + this.id );
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.push();

      var gl = this.gl;

      if ( this.dirty && !this.disposed ) {
        this.dirty = false;

        // update drawables, so that they have vertex arrays up to date, etc.
        while ( this.dirtyDrawables.length ) {
          this.dirtyDrawables.pop().update();
        }

        // ensure sprite sheet textures are up-to-date
        var numSpriteSheets = this.spriteSheets.length;
        for ( var i = 0; i < numSpriteSheets; i++ ) {
          this.spriteSheets[i].updateTexture();
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
        this.projectionMatrix.rowMajor( 2 / this.display.width, 0, -1,
                                        0, -2 / this.display.height, 1,
                                        0, 0, 1 );

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

      // will trigger changes to the spritesheets for images, or initialization for others
      drawable.onAddToBlock( this );
    },

    removeDrawable: function( drawable ) {
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( '#' + this.id + '.removeDrawable ' + drawable.toString() );

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
        var spriteSheet = this.spriteSheets[i];
        sprite = spriteSheet.addImage( image, width, height );
        if ( sprite ) {
          break;
        }
      }
      if ( !sprite ) {
        var newSpriteSheet = new SpriteSheet( true ); // use mipmaps for now?
        sprite = newSpriteSheet.addImage( image, width, height );
        newSpriteSheet.initializeContext( this.gl );
        newSpriteSheet.createTexture();
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
  *----------------------------------------------------------------------------*/

  // TODO: Processor super-type?

  WebGLBlock.CustomProcessor = function( webglBlock ) {
    this.webglBlock = webglBlock;

    this.drawable = null;
  };
  inherit( Object, WebGLBlock.CustomProcessor, {
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
        this.drawable.draw();
        this.drawCount++;
        this.drawable = null;
      }
    }
  } );

  WebGLBlock.TexturedTrianglesProcessor = function( webglBlock ) {
    this.webglBlock = webglBlock;
    var gl = this.gl = webglBlock.gl;

    assert && assert( webglBlock.gl );
    this.shaderProgram = new ShaderProgram( gl, [
      // vertex shader
      'attribute vec4 aVertex;',
      // 'attribute vec2 aTextureCoord;',
      'varying vec2 vTextureCoord;',
      'uniform mat3 uProjectionMatrix;',

      'void main() {',
      '  vTextureCoord = aVertex.zw;',
      '  vec3 ndc = uProjectionMatrix * vec3( aVertex.xy, 1.0 );', // homogeneous map to to normalized device coordinates
      '  gl_Position = vec4( ndc.xy, 0.0, 1.0 );',
      '}'
    ].join( '\n' ), [
      // fragment shader
      'precision mediump float;',
      'varying vec2 vTextureCoord;',
      'uniform sampler2D uTexture;',

      'void main() {',
      '  gl_FragColor = texture2D( uTexture, vTextureCoord, -0.7 );', // mipmap LOD bias of -0.7 (for now)
      '}'
    ].join( '\n' ), {
      // attributes: [ 'aVertex', 'aTextureCoord' ],
      attributes: [ 'aVertex' ],
      uniforms: [ 'uTexture', 'uProjectionMatrix' ]
    } );

    this.vertexBuffer = gl.createBuffer();
    this.lastArrayLength = 128; // initial vertex buffer array length
    this.vertexArray = new Float32Array( this.lastArrayLength );

    gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
    gl.bufferData( gl.ARRAY_BUFFER, this.vertexArray, gl.DYNAMIC_DRAW ); // fully buffer at the start
  };
  inherit( Object, WebGLBlock.TexturedTrianglesProcessor, {
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
      gl.uniformMatrix3fv( this.shaderProgram.uniformLocations.uProjectionMatrix, false, this.webglBlock.projectionMatrixArray );

      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
      // if we increased in length, we need to do a full bufferData to resize it on the GPU side
      if ( this.vertexArray.length > this.lastArrayLength ) {
        gl.bufferData( gl.ARRAY_BUFFER, this.vertexArray, gl.DYNAMIC_DRAW ); // fully buffer at the start
      }
      // otherwise do a more efficient update that only sends part of the array over
      else {
        gl.bufferSubData( gl.ARRAY_BUFFER, 0, this.vertexArray.subarray( 0, this.vertexArrayIndex ) );
      }
      gl.vertexAttribPointer( this.shaderProgram.attributeLocations.aVertex, 4, gl.FLOAT, false, 0, 0 );
      // TODO: test striping
      // var sizeOfFloat = 4;
      // gl.vertexAttribPointer( this.shaderProgram.attributeLocations.aVertex, 2, gl.FLOAT, false, 4 * sizeOfFloat, 0 * sizeOfFloat );
      // gl.vertexAttribPointer( this.shaderProgram.attributeLocations.aTextureCoord, 2, gl.FLOAT, false, 4 * sizeOfFloat, 2 * sizeOfFloat );

      gl.activeTexture( gl.TEXTURE0 );
      gl.bindTexture( gl.TEXTURE_2D, this.currentSpriteSheet.texture );
      gl.uniform1i( this.shaderProgram.uniformLocations.uTexture, 0 );

      gl.drawArrays( gl.TRIANGLES, 0, this.vertexArrayIndex / 4 );

      gl.bindTexture( gl.TEXTURE_2D, null );

      this.drawCount++;

      this.currentSpriteSheet = null;
      this.vertexArrayIndex = 0;
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
