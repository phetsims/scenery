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
  
  scenery.LayerType = function LayerType( Constructor, name, bitmask, renderer, args ) {
    this.Constructor = Constructor;
    this.name = name;
    this.bitmask = bitmask;
    this.renderer = renderer;
    this.args = args;
  };
  var LayerType = scenery.LayerType;
  
  LayerType.prototype = {
    constructor: LayerType,
    
    supportsRenderer: function( renderer ) {
      // NOTE: if this is changed off of instance equality, update supportsNode below
      return this.renderer === renderer;
    },
    
    supportsBitmask: function( bitmask ) {
      return ( this.bitmask & bitmask ) !== 0;
    },
    
    supportsNode: function( node ) {
      // for now, only check the renderer that we are interested in
      return node.supportsRenderer( this.renderer );
    },
    
    createLayer: function( args ) {
      var Constructor = this.Constructor;
      return new Constructor( _.extend( {}, args, this.args ) ); // allow overriding certain arguments if necessary by the LayerType
    }
  };
  
  return LayerType;
} );


