// Copyright 2002-2013, University of Colorado

/**
 * Contains information about what renderers (and a few other flags) are supported for an entire subtree.
 *
 * We effectively do this by tracking bitmask changes from scenery.js (used for rendering properties in general). In particular, we count
 * how many zeros in the bitmask we have in key places.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var scenery = require( 'SCENERY/scenery' );
  // require( 'SCENERY/layers/Renderer' );
  
  var bits = [
    scenery.bitmaskSupportsCanvas,
    scenery.bitmaskSupportsSVG,
    scenery.bitmaskSupportsDOM,
    scenery.bitmaskSupportsWebGL,
    scenery.bitmaskNotPainted,
    scenery.bitmaskBoundsValid
  ];
  var numBits = bits.length;
  
  scenery.RendererSummary = function RendererSummary( node ) {
    // NOTE: assumes that we are created in the Node constructor
    assert && assert( node._rendererBitmask === scenery.bitmaskNodeDefault, 'Node must have a default bitmask when creating a RenderSummary' );
    assert && assert( node._children.length === 0, 'Node cannot have children when creating a RenderSummary' );
    
    this.node = node;
    
    // initialize all of the defaults
    for ( var i = 0; i < numBits; i++ ) {
      var bit = bits[i];
      // we count the number of 0s
      this[bit] = ( scenery.bitmaskNodeDefault & bit ) === 0 ? 1 : 0;
    }
    
    this.bitmask = this.computeBitmask();
  };
  var RendererSummary = scenery.RendererSummary;
  
  RendererSummary.prototype = {
    constructor: RendererSummary,
    
    computeBitmask: function() {
      var bitmask = 0;
      for ( var i = 0; i < numBits; i++ ) {
        var bit = bits[i];
        
        // remember, if the count is zero, the bit is set
        if ( !this[bit] ) {
          bitmask |= bit;
        }
      }
      return bitmask;
    },
    
    bitIncrement: function( bit ) {
      var newCount = ++this[bit];
      if ( newCount === 1 ) {
        // if the count goes from 0 to 1, it means our combined bit went from 1 to 0 (reversed)
        this.notifyBitUnset( bit );
      }
    },
    
    bitDecrement: function( bit ) {
      var newCount = --this[bit];
      assert && assert( newCount >= 0, 'bitcount always needs to be above 0' );
      if ( newCount === 0 ) {
        // if the count goes from 1 to 0, it means our combined bit went from 0 to 1 (reversed)
        this.notifyBitSet( bit );
      }
    },
    
    notifyBitSet: function( bit ) {
      this.bitmask = this.computeBitmask();
      
      var len = this.node._parents.length;
      for ( var i = 0; i < len; i++ ) {
        this.node._parents[i]._rendererSummary.bitDecrement( bit );
      }
    },
    
    notifyBitUnset: function( bit ) {
      this.bitmask = this.computeBitmask();
      
      var len = this.node._parents.length;
      for ( var i = 0; i < len; i++ ) {
        this.node._parents[i]._rendererSummary.bitIncrement( bit );
      }
    },
    
    // use a bitmask of all 1s to represent 'does not exist' since we count zeros
    bitmaskChange: function( oldBitmask, newBitmask ) {
      var changeBitmask = oldBitmask ^ newBitmask;
      
      for ( var i = 0; i < numBits; i++ ) {
        var bit = bits[i];
        if ( ( bit & changeBitmask ) !== 0 ) {
          var currentValue = bit & newBitmask;
          
          if ( currentValue !== 0 ) {
            // we set the bit (used to be 0, now is 1)
            this.bitDecrement( bit );
          } else {
            // we unset the bit (used to be 1, now is 0)
            this.bitIncrement( bit );
          }
        }
      }
    }
  };
  
  return RendererSummary;
} );
