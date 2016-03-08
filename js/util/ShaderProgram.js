// Copyright 2014-2015, University of Colorado Boulder

/**
 * Abstraction over the shader program
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Util = require( 'SCENERY/util/Util' );

  function ShaderProgram( gl, vertexSource, fragmentSource, options ) {
    options = _.extend( {
      attributes: [], // {Array.<string>} (vertex) attribute names in the shader source
      uniforms: [] // {Array.<string>} uniform names in the shader source
    }, options );

    // store parameters so that we can recreate the shader program on context loss
    this.vertexSource = vertexSource;
    this.fragmentSource = fragmentSource;
    this.attributeNames = options.attributes;
    this.uniformNames = options.uniforms;

    this.initialize( gl );
  }

  scenery.register( 'ShaderProgram', ShaderProgram );

  return inherit( Object, ShaderProgram, {
    // initializes (or reinitializes) the WebGL state and uniform/attribute references.
    initialize: function( gl ) {
      var self = this;
      this.gl = gl; // TODO: create them with separate contexts

      this.used = false;

      this.program = this.gl.createProgram();

      this.vertexShader = Util.createShader( this.gl, this.vertexSource, this.gl.VERTEX_SHADER );
      this.fragmentShader = Util.createShader( this.gl, this.fragmentSource, this.gl.FRAGMENT_SHADER );

      this.gl.attachShader( this.program, this.vertexShader );
      this.gl.attachShader( this.program, this.fragmentShader );

      this.gl.linkProgram( this.program );

      if ( !this.gl.getProgramParameter( this.program, this.gl.LINK_STATUS ) ) {
        console.log( 'GLSL link error:' );
        console.log( this.gl.getProgramInfoLog( this.program ) );
        console.log( 'for vertex shader' );
        console.log( this.vertexSource );
        console.log( 'for fragment shader' );
        console.log( this.fragmentSource );

        // Normally it would be best to throw an exception here, but a context loss could cause the shader parameter check
        // to fail, and we must handle context loss gracefully between any adjacent pair of gl calls.
        // Therefore, we simply report the errors to the console.  See #279
      }

      // clean these up, they aren't needed after the link
      this.gl.deleteShader( this.vertexShader );
      this.gl.deleteShader( this.fragmentShader );

      this.uniformLocations = {}; // map name => uniform location for program
      this.attributeLocations = {}; // map name => attribute location for program
      this.activeAttributes = {}; // map name => boolean (enabled)

      _.each( this.attributeNames, function( attributeName ) {
        self.attributeLocations[ attributeName ] = self.gl.getAttribLocation( self.program, attributeName );
        self.activeAttributes[ attributeName ] = true; // default to enabled
      } );
      _.each( this.uniformNames, function( uniformName ) {
        self.uniformLocations[ uniformName ] = self.gl.getUniformLocation( self.program, uniformName );
      } );

      this.isInitialized = true;
    },

    use: function() {
      if ( this.used ) { return; }

      var self = this;

      this.used = true;

      this.gl.useProgram( this.program );

      // enable the active attributes
      _.each( this.attributeNames, function( attributeName ) {
        if ( self.activeAttributes[ attributeName ] ) {
          self.enableVertexAttribArray( attributeName );
        }
      } );
    },

    activateAttribute: function( attributeName ) {
      // guarded so we don't enable twice
      if ( !this.activeAttributes[ attributeName ] ) {
        this.activeAttributes[ attributeName ] = true;

        if ( this.used ) {
          this.enableVertexAttribArray( attributeName );
        }
      }
    },

    enableVertexAttribArray: function( attributeName ) {
      this.gl.enableVertexAttribArray( this.attributeLocations[ attributeName ] );
    },

    unuse: function() {
      if ( !this.used ) { return; }

      var self = this;

      this.used = false;

      _.each( this.attributeNames, function( attributeName ) {
        if ( self.activeAttributes[ attributeName ] ) {
          self.disableVertexAttribArray( attributeName );
        }
      } );
    },

    disableVertexAttribArray: function( attributeName ) {
      this.gl.disableVertexAttribArray( this.attributeLocations[ attributeName ] );
    },

    deactivateAttribute: function( attributeName ) {
      // guarded so we don't disable twice
      if ( this.activeAttributes[ attributeName ] ) {
        this.activeAttributes[ attributeName ] = false;

        if ( this.used ) {
          this.disableVertexAttribArray( attributeName );
        }
      }
    },

    dispose: function() {
      this.gl.deleteProgram( this.program );
    }
  } );
} );
