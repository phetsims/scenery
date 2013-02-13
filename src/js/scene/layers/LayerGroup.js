// Copyright 2002-2012, University of Colorado

/**
 * Handles a group of layers that represent all applicable layers under
 * a root node.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

var scenery = scenery || {};

(function(){
  "use strict";
  
  scenery.LayerGroup = function( root ) {
    this.layers = [];
    this.root = root;
    
    // TODO: add tree form for optimization
  };
  var LayerGroup = scenery.LayerGroup;
  
  LayerGroup.prototype = {
    constructor: LayerGroup,
    
    isEmpty: function() {
      return this.layers.length === 0;
    },
    
    layerLookup: function( path ) {
      phet.assert( !( path.isEmpty() || path.nodes[0] !== this.root ), 'layerLookup root matches' );
      
      if ( this.isEmpty() ) {
        return null;
      }
      
      for ( var i = 0; i < this.layers.length; i++ ) {
        var layer = this.layers[i];
        if ( path.compare( layer.endPath ) !== 1 ) {
          return layer;
        }
      }
      
      throw new Error( 'node not contained in a layer' );
    }
  };
  
})();
