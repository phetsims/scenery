// Copyright 2002-2013, University of Colorado

/**
 * WebGL-backed layer
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Matrix4 = require( 'DOT/Matrix4' );

  var scenery = require( 'SCENERY/scenery' );

  var Layer = require( 'SCENERY/layers/Layer' ); // uses Layer's prototype for inheritance
  require( 'SCENERY/util/Trail' );
  require( 'SCENERY/util/TrailPointer' );
  require( 'SCENERY/util/Util' );
  var ShaderProgram = require( 'SCENERY/util/ShaderProgram' );

  //Scenery uses Matrix3 and WebGL uses Matrix4, so we must convert.
  function matrix3To4( matrix3 ) {
    return new Matrix4(
      matrix3.m00(), matrix3.m01(), 0, matrix3.m02(),
      matrix3.m10(), matrix3.m11(), 0, matrix3.m12(),
      0, 0, 1, 0,
      0, 0, 0, 1 );
  }

  /**
   * Constructor for WebGLLayer
   * @param args renderer options (none at the moment)
   * @constructor
   */
  scenery.WebGLLayer = function WebGLLayer( args ) {
    sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' constructor' );
    Layer.call( this, args );

    this.dirty = true;

    // TODO: deprecate Scene's backing scale, and handle this on a layer-by-layer option?
    this.backingScale = args.scene.backingScale;
    if ( args.fullResolution !== undefined ) {
      this.backingScale = args.fullResolution ? scenery.Util.backingScale( document.createElement( 'canvas' ).getContext( '2d' ) ) : 1;
    }

    this.logicalWidth = this.scene.sceneBounds.width;
    this.logicalHeight = this.scene.sceneBounds.height;

    this.canvas = document.createElement( 'canvas' );

    this.canvas.width = this.logicalWidth * this.backingScale;
    this.canvas.height = this.logicalHeight * this.backingScale;
    this.canvas.style.width = this.logicalWidth + 'px';
    this.canvas.style.height = this.logicalHeight + 'px';
    this.canvas.style.position = 'absolute';
    this.canvas.style.left = '0';
    this.canvas.style.top = '0';

    // add this layer on top (importantly, the constructors of the layers are called in order)
    this.$main.append( this.canvas );

    this.scene = args.scene;

    this.isWebGLLayer = true;

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

    this.initialize();

    this.instances = [];
  };
  var WebGLLayer = scenery.WebGLLayer;

  inherit( Layer, WebGLLayer, {
      initialize: function() {
        var gl = this.gl;
        gl.clearColor( 0.0, 0.0, 0.0, 0.0 );

        gl.enable( gl.BLEND );
        gl.blendFunc( gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA );

        //This is an ubershader, which handles all of the different vertex/fragment types in a single shader
        //To reduce overhead of switching programs.
        //TODO: Perhaps the shader program should be loaded through an external file with a RequireJS plugin
        this.shaderProgram = new ShaderProgram( gl,

          /********** Vertex Shader **********/

          //The vertex to be transformed
            'attribute vec3 aVertex;\n' +

            // The transformation matrix
            'uniform mat4 uMatrix;\n' +

            // The texture coordinates (if any)
            //TODO: Is this needed here in the vertex shader?
            'varying vec2 texCoord;\n' +

            // The color to render (if any)
            //TODO: Is this needed here in the vertex shader?
            'uniform vec4 uColor;\n' +
            'void main() {\n' +

            //This texture is not needed for rectangles, but we (JO/SR) don't expect it to be expensive, so we leave
            //it for simplicity
            '  texCoord = aVertex.xy;\n' +
            '  gl_Position = uMatrix * vec4( aVertex, 1 );\n' +
            '}',

          /********** Fragment Shader **********/

          //Directive to indicate high precision
            'precision highp float;\n' +

            //Texture coordinates (for images)
            'varying vec2 texCoord;\n' +

            //Color (rgba) for filled items
            'uniform vec4 uColor;\n' +

            //Fragment type such as fragmentTypeFill or fragmentTypeTexture
            'uniform int uFragmentType;\n' +

            //Texture (if any)
            'uniform sampler2D uTexture;\n' +
            'void main() {\n' +
            '  if (uFragmentType==' + WebGLLayer.fragmentTypeFill + '){\n' +
            '    gl_FragColor = uColor;\n' +
            '  }else if (uFragmentType==' + WebGLLayer.fragmentTypeTexture + '){\n' +
            '    gl_FragColor = texture2D( uTexture, texCoord );\n' +
            '  }\n' +
            '}',

          ['aVertex'], // attribute names
          ['uTexture', 'uMatrix', 'uColor', 'uFragmentType'] // uniform names
        );

        this.setSize( this.logicalWidth, this.logicalHeight );

        this.shaderProgram.use();
      },

      render: function( scene, args ) {
        var gl = this.gl;

        if ( this.dirty ) {
          gl.clear( this.gl.COLOR_BUFFER_BIT );

          // (0,height) => (0, -2) => ( 1, -1 )

          var projectionMatrix = Matrix4.translation( -1, 1, 0 ).timesMatrix( Matrix4.scaling( 2 / this.logicalWidth, -2 / this.logicalHeight, 1 ) );

          var length = this.instances.length;
          for ( var i = 0; i < length; i++ ) {
            var instance = this.instances[i];

            if ( instance.trail.isVisible() ) {
              // TODO: this is expensive overhead!
              var modelViewMatrix = matrix3To4( instance.trail.getMatrix() );

              instance.data.drawable.render( this.shaderProgram, projectionMatrix.timesMatrix( modelViewMatrix ) );
            }
          }
        }
      },

      switchToProgram: function( shaderProgram ) {
        if ( shaderProgram !== this.shaderProgram ) {
          this.shaderProgram && this.shaderProgram.unuse();
          shaderProgram.use();

          this.shaderProgram = shaderProgram;
        }
      },

      setSize: function( width, height ) {
        this.gl.viewport( 0, 0, width, height );
      },

      dispose: function() {
        Layer.prototype.dispose.call( this );

        this.canvas.parentNode.removeChild( this.canvas );

        this.shaderProgram.unuse();
        this.shaderProgram.dispose();
      },

      applyTransformationMatrix: function( matrix ) {

      },

      // returns next zIndex in place. allows layers to take up more than one single zIndex
      reindex: function( zIndex ) {
        Layer.prototype.reindex.call( this, zIndex );

        if ( this.zIndex !== zIndex ) {
          this.canvas.style.zIndex = zIndex;
          this.zIndex = zIndex;
        }
        return zIndex + 1;
      },

      pushClipShape: function( shape ) {

      },

      popClipShape: function() {

      },

      getSVGString: function() {
        // TODO: probably broken
        return '<image xmlns:xlink="' + scenery.xlinkns + '" xlink:href="' + this.canvas.toDataURL() + '" x="0" y="0" height="' + this.canvas.height + 'px" width="' + this.canvas.width + 'px"/>';
      },

      // TODO: note for DOM we can do https://developer.mozilla.org/en-US/docs/HTML/Canvas/Drawing_DOM_objects_into_a_canvas
      renderToCanvas: function( canvas, context, delayCounts ) {
        context.drawImage( this.canvas, 0, 0 );
      },

      addInstance: function( instance ) {
        var trail = instance.trail;

        sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' addInstance: ' + trail.toString() );
        Layer.prototype.addInstance.call( this, instance );

        instance.data.drawable = instance.node.createWebGLDrawable( this.gl );

        // insert into this.instances array
        var added = false;
        for ( var i = 0; i < this.instances.length; i++ ) {
          if ( instance.trail.compare( this.instances[i].trail ) < 0 ) {
            this.instances.splice( i, 0, instance );
            added = true;
            break;
          }
        }
        if ( !added ) {
          this.instances.push( instance );
        }

        this.markWebGLDirty();
      },

      removeInstance: function( instance ) {
        var trail = instance.trail;

        sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' removeInstance: ' + trail.toString() );
        Layer.prototype.removeInstance.call( this, instance );

        instance.data.drawable.dispose();
        instance.data.drawable = null;

        // remove from this.instances array
        this.instances.splice( this.instances.indexOf( instance ), 1 );

        this.markWebGLDirty();
      },

      getName: function() {
        return 'webgl';
      },

      markWebGLDirty: function() {
        this.dirty = true;
      },

      /*---------------------------------------------------------------------------*
       * Events from Instances
       *----------------------------------------------------------------------------*/

      notifyVisibilityChange: function( instance ) {
        sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' notifyVisibilityChange: ' + instance.trail.toString() );
        // old paint taken care of in notifyBeforeSubtreeChange()

        this.markWebGLDirty();
      },

      notifyOpacityChange: function( instance ) {
        sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' notifyOpacityChange: ' + instance.trail.toString() );
        // old paint taken care of in notifyBeforeSubtreeChange()

        this.markWebGLDirty();
      },

      notifyClipChange: function( instance ) {
        sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' notifyClipChange: ' + instance.trail.toString() );
        // old paint taken care of in notifyBeforeSubtreeChange()

        this.markWebGLDirty();
      },

      // only a painted trail under this layer
      notifyBeforeSelfChange: function( instance ) {
        sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' notifyBeforeSelfChange: ' + instance.trail.toString() );

        this.markWebGLDirty();

        instance.node.updateWebGLDrawable( instance.data.drawable );
      },

      notifyBeforeSubtreeChange: function( instance ) {
        sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' notifyBeforeSubtreeChange: ' + instance.trail.toString() );

        this.markWebGLDirty();
      },

      // only a painted trail under this layer
      notifyDirtySelfPaint: function( instance ) {
        sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' notifyDirtySelfPaint: ' + instance.trail.toString() );

        this.markWebGLDirty();

        instance.node.updateWebGLDrawable( instance.data.drawable );
      },

      notifyDirtySubtreePaint: function( instance ) {
        sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' notifyDirtySubtreePaint: ' + instance.trail.toString() );

        this.markWebGLDirty();
      },

      notifyTransformChange: function( instance ) {
        // sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' notifyTransformChange: ' + instance.trail.toString() );
        // TODO: how best to mark this so if there are multiple 'movements' we don't get called more than needed?
        // this.canvasMarkSubtree( instance );

        this.markWebGLDirty();
      },

      // only a painted trail under this layer (for now)
      notifyBoundsAccuracyChange: function( instance ) {
        sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' notifyBoundsAccuracyChange: ' + instance.trail.toString() );

        this.markWebGLDirty();
      }
    },

    //Statics
    {
      fragmentTypeFill: 0,
      fragmentTypeTexture: 1
    } );

  return WebGLLayer;
} );