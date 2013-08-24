// Copyright 2002-2013, University of Colorado

/**
 * A description of layer settings and the ability to create a layer with those settings.
 * Used internally for the layer building process.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.LayerType = function LayerType( Constructor, name, renderer, args ) {
    this.Constructor = Constructor;
    this.name = name;
    this.renderer = renderer;
    this.args = args;
  };
  var LayerType = scenery.LayerType;
  
  LayerType.prototype = {
    constructor: LayerType,
    
    supportsRenderer: function( renderer ) {
      return this.renderer === renderer;
    },
    
    supportsNode: function( node ) {
      var supportedRenderers = node._supportedRenderers;
      var i = supportedRenderers.length;
      while ( i-- ) {
        if ( this.supportsRenderer( supportedRenderers[i] ) ) {
          return true;
        }
      }
      return false;
    },
    
    createLayer: function( args ) {
      var Constructor = this.Constructor;
      return new Constructor( _.extend( {}, args, this.args ) ); // allow overriding certain arguments if necessary by the LayerType
    }
  };
  
  return LayerType;
} );


