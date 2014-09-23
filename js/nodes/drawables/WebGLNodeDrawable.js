// Copyright 2002-2013, University of Colorado

/**
 * WebGL state for rendering WebGLNodes.  Delegates all work to the parent WebGLNode
 * TODO: Should this class be deleted (and let WebGLNode just do everything?)
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 * @author Sam Reid
 */
define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  scenery.WebGLNodeDrawable = function WebGLNodeDrawable( gl, webGLNode ) {
    this.webGLNode = webGLNode;

    this.gl = gl;

    //Do most of the work in an initialize function to handle WebGL context loss
    this.initialize( gl );
  };

  return inherit( Object, scenery.WebGLNodeDrawable, {
    initialize: function() {
      this.webGLNode.initialize( this.gl );
    },

    render: function( shaderProgram, viewMatrix ) {
      this.webGLNode.render( this.gl, shaderProgram, viewMatrix );
    },

    dispose: function() {
      this.webGLNode.dispose( this.gl );
    }
  } );
} );