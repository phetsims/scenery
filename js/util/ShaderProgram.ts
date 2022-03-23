// Copyright 2014-2022, University of Colorado Boulder

/**
 * Abstraction over the shader program
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import merge from '../../../phet-core/js/merge.js';
import { scenery, Utils } from '../imports.js';

export type ShaderProgramOptions = {
  // (vertex) attribute names in the shader source
  attributes?: string[];

  // uniform names in the shader source
  uniforms?: string[];
};

export default class ShaderProgram {

  // store parameters so that we can recreate the shader program on context loss
  private vertexSource: string;
  private fragmentSource: string;
  private attributeNames: string[];
  private uniformNames: string[];

  private gl!: WebGLRenderingContext;
  private used!: boolean;
  private program!: WebGLProgram;
  private vertexShader!: WebGLShader;
  private fragmentShader!: WebGLShader;

  uniformLocations!: { [ key: string ]: WebGLUniformLocation };
  attributeLocations!: { [ key: string ]: number };
  activeAttributes!: { [ key: string ]: boolean }; // whether they are enabled

  private isInitialized!: boolean;

  constructor( gl: WebGLRenderingContext, vertexSource: string, fragmentSource: string, providedOptions?: ShaderProgramOptions ) {
    const options = merge( {
      attributes: [],
      uniforms: []
    }, providedOptions );

    this.vertexSource = vertexSource;
    this.fragmentSource = fragmentSource;
    this.attributeNames = options.attributes;
    this.uniformNames = options.uniforms;

    this.initialize( gl );
  }

  /**
   * Initializes (or reinitializes) the WebGL state and uniform/attribute references.
   */
  initialize( gl: WebGLRenderingContext ) {
    // @private {WebGL2RenderingContext}
    this.gl = gl; // TODO: create them with separate contexts

    // @private {boolean}
    this.used = false;

    // @private {WebGLProgram}
    this.program = this.gl.createProgram()!;

    // @private {WebGLShader}
    this.vertexShader = Utils.createShader( this.gl, this.vertexSource, this.gl.VERTEX_SHADER );
    this.fragmentShader = Utils.createShader( this.gl, this.fragmentSource, this.gl.FRAGMENT_SHADER );

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

    // @public {Object}
    this.uniformLocations = {}; // map name => uniform location for program
    this.attributeLocations = {}; // map name => attribute location for program
    this.activeAttributes = {}; // map name => boolean (enabled)

    _.each( this.attributeNames, attributeName => {
      this.attributeLocations[ attributeName ] = this.gl.getAttribLocation( this.program, attributeName );
      this.activeAttributes[ attributeName ] = true; // default to enabled
    } );
    _.each( this.uniformNames, uniformName => {
      this.uniformLocations[ uniformName ] = this.gl.getUniformLocation( this.program, uniformName )!;
    } );

    // @private {boolean}
    this.isInitialized = true;
  }

  /**
   * @public
   */
  use() {
    if ( this.used ) { return; }

    this.used = true;

    this.gl.useProgram( this.program );

    // enable the active attributes
    _.each( this.attributeNames, attributeName => {
      if ( this.activeAttributes[ attributeName ] ) {
        this.enableVertexAttribArray( attributeName );
      }
    } );
  }

  activateAttribute( attributeName: string ) {
    // guarded so we don't enable twice
    if ( !this.activeAttributes[ attributeName ] ) {
      this.activeAttributes[ attributeName ] = true;

      if ( this.used ) {
        this.enableVertexAttribArray( attributeName );
      }
    }
  }

  enableVertexAttribArray( attributeName: string ) {
    this.gl.enableVertexAttribArray( this.attributeLocations[ attributeName ] );
  }

  unuse() {
    if ( !this.used ) { return; }

    this.used = false;

    _.each( this.attributeNames, attributeName => {
      if ( this.activeAttributes[ attributeName ] ) {
        this.disableVertexAttribArray( attributeName );
      }
    } );
  }

  disableVertexAttribArray( attributeName: string ) {
    this.gl.disableVertexAttribArray( this.attributeLocations[ attributeName ] );
  }

  deactivateAttribute( attributeName: string ) {
    // guarded so we don't disable twice
    if ( this.activeAttributes[ attributeName ] ) {
      this.activeAttributes[ attributeName ] = false;

      if ( this.used ) {
        this.disableVertexAttribArray( attributeName );
      }
    }
  }

  /**
   * Releases references
   */
  dispose() {
    this.gl.deleteProgram( this.program );
  }
}

scenery.register( 'ShaderProgram', ShaderProgram );
