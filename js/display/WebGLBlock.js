// Copyright 2002-2014, University of Colorado Boulder

/**
 * Handles a visual WebGL layer of drawables.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Sam Reid
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var Matrix4 = require( 'DOT/Matrix4' );
  var scenery = require( 'SCENERY/scenery' );
  var FittedBlock = require( 'SCENERY/display/FittedBlock' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var Util = require( 'SCENERY/util/Util' );
  var ShaderProgram = require( 'SCENERY/util/ShaderProgram' );

  scenery.WebGLBlock = function WebGLBlock( display, renderer, transformRootInstance, filterRootInstance ) {
    this.initialize( display, renderer, transformRootInstance, filterRootInstance );
  };
  var WebGLBlock = scenery.WebGLBlock;

  inherit( FittedBlock, WebGLBlock, {
    initialize: function( display, renderer, transformRootInstance, filterRootInstance ) {
      var block = this;

      this.initializeFittedBlock( display, renderer, transformRootInstance );

      this.filterRootInstance = filterRootInstance;

      this.dirtyDrawables = cleanArray( this.dirtyDrawables );

      if ( !this.domElement ) {
        //OHTWO TODO: support tiled WebGL handling (will need to wrap then in a div, or something)
        this.canvas = document.createElement( 'canvas' );

        // If the display was instructed to make a WebGL context that can simulate context loss, wrap it here, see #279
        if ( this.display.options.webglMakeLostContextSimulatingCanvas ) {
          this.canvas = window.WebGLDebugUtils.makeLostContextSimulatingCanvas( this.canvas );
        }

        this.canvas.style.position = 'absolute';
        this.canvas.style.left = '0';
        this.canvas.style.top = '0';
        this.canvas.style.pointerEvents = 'none';

        this.gl = null;
        try {
          this.gl = this.canvas.getContext( 'webgl' ) || this.canvas.getContext( 'experimental-webgl' );
          // TODO: check for required extensions
        }
        catch( e ) {
          // TODO: handle gracefully
          throw e;
        }
        if ( !this.gl ) {
          throw new Error( 'Unable to load WebGL' );
        }

        this.domElement = this.canvas;

      }

      this.projectionMatrix = this.projectionMatrix || new Matrix4();

      // Keep track of whether the context is lost, so that we can avoid trying to render while the context is lost.
      this.webglContextIsLost = false;

      // Callback for context loss, see #279
      this.canvas.addEventListener( 'webglcontextlost', function( event ) {
        console.log( 'context lost' );

        // khronos does not explain why we must prevent default in webgl context loss, but we must do so:
        // http://www.khronos.org/webgl/wiki/HandlingContextLost#Handling_Lost_Context_in_WebGL
        event.preventDefault();
        block.webglContextIsLost = true;
      }, false );

      // Only used when display.options.webglContextLossIncremental is defined
      var numCallsToLoseContext = 1;

      // Callback for context restore, see #279
      this.canvas.addEventListener( 'webglcontextrestored', function( event ) {
        console.log( 'context restored' );
        block.webglContextIsLost = false;

        // When context is restored, optionally simulate another context loss at an increased number of gl calls
        // This is because we must test for context loss between every pair of gl calls
        if ( display.options.webglContextLossIncremental ) {
          console.log( 'simulating context loss in ', numCallsToLoseContext, 'gl calls.' );
          block.canvas.loseContextInNCalls( numCallsToLoseContext );
          numCallsToLoseContext++;
        }

        // Reinitialize the layer state
        block.initializeWebGLState();

        // Reinitialize the webgl state for every instance's drawable
        for ( var drawable = this.firstDrawable; drawable !== null; drawable = drawable.nextDrawable ) {
          drawable.initializeWebGLState();

          if ( drawable === this.lastDrawable ) { break; }
        }

        // Mark for repainting
        block.markDirty();
      }, false );


      // reset any fit transforms that were applied
      Util.prepareForTransform( this.canvas, this.forceAcceleration );
      Util.unsetTransform( this.canvas ); // clear out any transforms that could have been previously applied

      // store our backing scale so we don't have to look it up while fitting
      this.backingScale = ( renderer & Renderer.bitmaskWebGLLowResolution ) ? 1 : scenery.Util.backingScale( this.gl );

      this.initializeWebGLState();

      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( 'initialized #' + this.id );
      // TODO: dirty list of nodes (each should go dirty only once, easier than scanning all?)

      return this;
    },

    initializeWebGLState: function() {
      var gl = this.gl;
      gl.clearColor( 0.0, 0.0, 0.0, 0.0 );

      gl.enable( gl.BLEND );
      gl.blendFunc( gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA );

      //This is an ubershader, which handles all of the different vertex/fragment types in a single shader
      //To reduce overhead of switching programs.
      //TODO: Perhaps the shader program should be loaded through an external file with a RequireJS plugin
      this.shaderProgram = new ShaderProgram( gl,

        /********** Vertex Shader **********/

          'precision mediump float;\n' +

          //The vertex to be transformed
          'attribute vec3 aVertex;\n' +

          // The texture coordinate
          'attribute vec2 aTexCoord;\n' +

          // The projection matrix
          'uniform mat4 uProjectionMatrix;\n' +

          // The model-view matrix
          'uniform mat4 uModelViewMatrix;\n' +

          // The texture coordinates (if any)
          //TODO: Is this needed here in the vertex shader?
          'varying vec2 texCoord;\n' +

          // The color to render (if any)
          //TODO: Is this needed here in the vertex shader?
          'uniform vec4 uColor;\n' +
          'void main() {\n' +

          //This texture is not needed for rectangles, but we (JO/SR) don't expect it to be expensive, so we leave
          //it for simplicity
          '  texCoord = aTexCoord;\n' +
          '  gl_Position = uProjectionMatrix * uModelViewMatrix * vec4( aVertex, 1 );\n' +
          '}',

        /********** Fragment Shader **********/

        //Directive to indicate high precision
          'precision mediump float;\n' +

          //Texture coordinates (for images)
          'varying vec2 texCoord;\n' +

          //Color (rgba) for filled items
          'uniform vec4 uColor;\n' +

          //Fragment type such as fragmentTypeFill or fragmentTypeTexture
          'uniform int uFragmentType;\n' +

          //Texture (if any)
          'uniform sampler2D uTexture;\n' +
          'void main() {\n' +
          '  if (uFragmentType==' + WebGLBlock.fragmentTypeFill + '){\n' +
          '    gl_FragColor = uColor;\n' +
          '  }else if (uFragmentType==' + WebGLBlock.fragmentTypeTexture + '){\n' +
          '    gl_FragColor = texture2D( uTexture, texCoord );\n' +
          '  }\n' +
          '}',

        ['aVertex', 'aTexCoord'], // attribute names
        ['uTexture', 'uProjectionMatrix', 'uModelViewMatrix', 'uColor', 'uFragmentType'] // uniform names
      );

      this.shaderProgram.deactivateAttribute( 'aVertex' );
      this.shaderProgram.deactivateAttribute( 'aTexCoord' );

      this.shaderProgram.use();
    },

    setSizeFullDisplay: function() {
      var size = this.display.getSize();
      this.canvas.width = size.width * this.backingScale;
      this.canvas.height = size.height * this.backingScale;
      this.canvas.style.width = size.width + 'px';
      this.canvas.style.height = size.height + 'px';
      this.updateWebGLDimension( 0, 0, size.width, size.height );
    },

    setSizeFitBounds: function() {
      var x = this.fitBounds.minX;
      var y = this.fitBounds.minY;
      //OHTWO TODO PERFORMANCE: see if we can get a speedup by putting the backing scale in our transform instead of with CSS?
      Util.setTransform( 'matrix(1,0,0,1,' + x + ',' + y + ')', this.canvas, this.forceAcceleration ); // reapply the translation as a CSS transform
      this.canvas.width = this.fitBounds.width * this.backingScale;
      this.canvas.height = this.fitBounds.height * this.backingScale;
      this.canvas.style.width = this.fitBounds.width + 'px';
      this.canvas.style.height = this.fitBounds.height + 'px';
      this.updateWebGLDimension( -x, -y, this.fitBounds.width, this.fitBounds.height );
    },

    updateWebGLDimension: function( x, y, width, height ) {
      this.gl.viewport( 0, 0, width * this.backingScale, height * this.backingScale );

      // (0,width) => (0, -2) => (-1, 1)
      // (0,height) => (0, -2) => ( 1, -1 )
      this.projectionMatrix.set( Matrix4.translation( -1, 1, 0 ).timesMatrix( Matrix4.scaling( 2 / width, -2 / height, 1 ) )
                                                                .timesMatrix( Matrix4.translation( x, y, 0 ) ) );
    },

    update: function() {
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.WebGLBlock( 'update #' + this.id );
      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.push();

      if ( this.webglContextIsLost ) {
        return;
      }
      var gl = this.gl;

      if ( this.dirty && !this.disposed ) {
        this.dirty = false;

        while ( this.dirtyDrawables.length ) {
          this.dirtyDrawables.pop().update();
        }

        // udpate the fit BEFORE drawing, since it may change our offset
        this.updateFit();

        gl.clear( this.gl.COLOR_BUFFER_BIT );

        gl.uniformMatrix4fv( this.shaderProgram.uniformLocations.uProjectionMatrix, false, this.projectionMatrix.entries );

        //OHTWO TODO: PERFORMANCE: create an array for faster drawable iteration (this is probably a hellish memory access pattern)
        for ( var drawable = this.firstDrawable; drawable !== null; drawable = drawable.nextDrawable ) {
          //OHTWO TODO: Performance for this lookup?
          // enable and required attributes, and disable the rest, since if we leave an attribute enabled and our
          // code doesn't use vertexAttribPointer to set it, WebGL will crash.
          for ( var i = 0; i < this.shaderProgram.attributeNames.length; i++ ) {
            var attributeName = this.shaderProgram.attributeNames[i];
            if ( _.contains( drawable.shaderAttributes, attributeName ) ) {
              this.shaderProgram.activateAttribute( attributeName );
            }
            else {
              this.shaderProgram.deactivateAttribute( attributeName );
            }
          }

          drawable.render( this.shaderProgram );

          if ( drawable === this.lastDrawable ) { break; }
        }
      }

      sceneryLog && sceneryLog.WebGLBlock && sceneryLog.pop();
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

      drawable.initializeContext( this.gl );
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

    // This method can be called to simulate context loss using the khronos webgl-debug context loss simulator, see #279
    simulateWebGLContextLoss: function() {
      console.log( 'simulating webgl context loss in WebGLBlock' );
      assert && assert( this.scene.webglMakeLostContextSimulatingCanvas );
      this.canvas.loseContextInNCalls( 5 );
    },

    toString: function() {
      return 'WebGLBlock#' + this.id + '-' + FittedBlock.fitString[this.fit];
    }
  }, {
    // Statics
    fragmentTypeFill: 0,
    fragmentTypeTexture: 1
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
