// Copyright 2002-2012, University of Colorado

/**
 * A description of layer settings and the ability to create a layer with those settings.
 * Used internally for the layer building process.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.LayerType = function( Constructor, name, renderer, args ) {
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
      var that = this;
      return _.some( node._supportedRenderers, function( renderer ) {
        return that.supportsRenderer( renderer );
      } );
    },
    
    createLayer: function( args, entry ) {
      var Constructor = this.Constructor;
      return new Constructor( _.extend( {}, args, this.args ), entry ); // allow overriding certain arguments if necessary by the LayerType
    }
  };
  
  return LayerType;
} );


