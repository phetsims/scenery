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
  
  scenery.LayerType = function( Constructor, name, backend, args ) {
    this.Constructor = Constructor;
    this.name = name;
    this.backend = backend;
    this.args = args;
  };
  var LayerType = scenery.LayerType;
  
  LayerType.prototype = {
    constructor: LayerType,
    
    supportsBackend: function( backend ) {
      return this.backend === backend;
    },
    
    supportsNode: function( node ) {
      var that = this;
      return _.some( node._supportedBackends, function( backend ) {
        return that.supportsBackend( backend );
      } );
    },
    
    createLayer: function( args, entry ) {
      var Constructor = this.Constructor;
      return new Constructor( _.extend( {}, args, this.args ), entry ); // allow overriding certain arguments if necessary by the LayerType
    }
  };
  
  return LayerType;
} );


