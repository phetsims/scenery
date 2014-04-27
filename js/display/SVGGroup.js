// Copyright 2002-2014, University of Colorado

/**
 * Poolable wrapper for SVG <group> elements. We store state and add listeners directly to the corresponding Node, so that we can set dirty flags and
 * smartly update only things that have changed. This takes a load off of SVGBlock.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var Poolable = require( 'PHET_CORE/Poolable' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.SVGGroup = function SVGGroup( block, instance, parent ) {
    this.initialize( block, instance, parent );
  };
  var SVGGroup = scenery.SVGGroup;
  
  SVGGroup.prototype = {
    constructor: SVGGroup,
    
    initialize: function( block, instance, parent ) {
      //OHTWO TODO: add collapsing groups! they can't have self drawables, transforms, filters, etc., and we probably shouldn't de-collapse groups
      
      this.block = block;
      this.instance = instance;
      this.node = instance.trail.lastNode();
      this.parent = parent;
      this.children = cleanArray( this.children );
      this.hasSelfDrawable = false;
      
      // general dirty flag (triggered on any other dirty event)
      this.dirty = true;
      
      // we won't listen for transform changes (or even want to set a transform) if our node is beneath a transform root
      this.willApplyTransforms = this.block.transformRootInstance.trail.nodes.length < this.instance.trail.nodes.length;
      
      // we won't listen for filter changes (or set filters, like opacity or visibility) if our node is beneath a filter root
      this.willApplyFilters = this.block.filterRootInstance.trail.nodes.length < this.instance.trail.nodes.length;
      
      // transform handling
      this.transformDirty = true;
      this.hasTransform = false;
      this.transformDirtyListener = this.transformDirtyListener || this.markTransformDirty.bind( this );
      if ( this.willApplyTransforms ) {
        this.node.onStatic( 'transform', this.transformDirtyListener );
      }
      
      // filter handling
      this.opacityDirty = true;
      this.visibilityDirty = true;
      this.clipDirty = true;
      this.hasOpacity = false;
      this.clipDefinition = null;
      this.clipPath = null;
      this.opacityDirtyListener = this.opacityDirtyListener || this.markOpacityDirty.bind( this );
      this.visibilityDirtyListener = this.visibilityDirtyListener || this.markVisibilityDirty.bind( this );
      this.clipDirtyListener = this.clipDirtyListener || this.markClipDirty.bind( this );
      if ( this.willApplyFilters ) {
        this.node.onStatic( 'opacity', this.opacityDirtyListener );
        this.node.onStatic( 'visibility', this.visibilityDirtyListener );
        this.node.onStatic( 'clip', this.clipDirtyListener );
      }
      
      // for tracking the order of child groups, we use a flag and update (reorder) once per updateDisplay if necessary.
      this.orderDirty = true;
      this.orderDirtyListener = this.orderDirtyListener || this.markOrderDirty.bind( this );
      this.node.onStatic( 'childInserted', this.orderDirtyListener );
      this.node.onStatic( 'childRemoved', this.orderDirtyListener );
      
      if ( !this.svgGroup ) {
        this.svgGroup = document.createElementNS( scenery.svgns, 'g' );
      }
      
      this.instance.addSVGGroup( this );
      
      this.block.markDirtyGroup( this ); // so we are marked and updated properly
    },
    
    addSelfDrawable: function( drawable ) {
      this.svgGroup.insertBefore( drawable.svgElement, this.children.length ? this.children[0].svgBlock : null );
      this.hasSelfDrawable = true;
    },
    
    removeSelfDrawable: function( drawable ) {
      this.hasSelfDrawable = false;
      this.svgGroup.removeChild( drawable.svgElement );
    },
    
    addChildGroup: function( group ) {
      this.markOrderDirty();
      
      group.parent = this;
      this.children.push( group );
      this.svgGroup.appendChild( group.svgGroup );
    },
    
    removeChildGroup: function( group ) {
      this.markOrderDirty();
      
      group.parent = null;
      this.children.splice( _.indexOf( this.children, group ), 1 );
      this.svgGroup.removeChild( group.svgGroup );
    },
    
    markDirty: function() {
      if ( !this.dirty ) {
        this.dirty = true;
        
        this.block.markDirtyGroup( this );
      }
    },
    
    /*---------------------------------------------------------------------------*
    * TODO: reduce filesize by creating these methods programatically. not done yet since I want to ensure correctness and make refactoring easier right now.
    *----------------------------------------------------------------------------*/
    
    markOrderDirty: function() {
      if ( !this.orderDirty ) {
        this.orderDirty = true;
        this.markDirty();
      }
    },
    
    markTransformDirty: function() {
      if ( !this.transformDirty ) {
        this.transformDirty = true;
        this.markDirty();
      }
    },
    
    markOpacityDirty: function() {
      if ( !this.opacityDirty ) {
        this.opacityDirty = true;
        this.markDirty();
      }
    },
    
    markVisibilityDirty: function() {
      if ( !this.visibilityDirty ) {
        this.visibilityDirty = true;
        this.markDirty();
      }
    },
    
    markClipDirty: function() {
      if ( !this.clipDirty ) {
        this.clipDirty = true;
        this.markDirty();
      }
    },
    
    update: function() {
      // we may have been disposed since being marked dirty on our block. we won't have a reference if we are disposed
      if ( !this.block ) {
        return;
      }
      
      var svgGroup = this.svgGroup;
      
      this.dirty = false;
      
      if ( this.willApplyTransforms ) {
        if ( this.transformDirty ) {
          this.transformDirty = false;
          
          var isIdentity = this.node.transform.isIdentity();
          
          if ( !isIdentity ) {
            this.hasTransform = true;
            svgGroup.setAttribute( 'transform', this.node.transform.getMatrix().getSVGTransform() );
          } else if ( this.hasTransform ) {
            this.hasTransform = false;
            svgGroup.removeAttribute( 'transform' );
          }
        }
      }
      
      if ( this.willApplyFilters ) {
        if ( this.visibilityDirty ) {
          this.visibilityDirty = false;
          
          svgGroup.style.display = this.node.isVisible() ? '' : 'none';
        }
        
        if ( this.opacityDirty ) {
          this.opacityDirty = false;
          
          if ( this.node.opacity !== 1 ) {
            this.hasOpacity = true;
            svgGroup.setAttribute( 'opacity', this.node.opacity );
          } else if ( this.hasOpacity ) {
            this.hasOpacity = false;
            svgGroup.removeAttribute( 'opacity' );
          }
        }
        
        if ( this.clipDirty ) {
          this.clipDirty = false;
          
          var clipId = 'clip' + this.node.getId();
          
          if ( this.node._clipArea ) {
            if ( !this.clipDefinition ) {
              this.clipDefinition = document.createElementNS( scenery.svgns, 'clipPath' );
              this.clipDefinition.setAttribute( 'id', clipId );
              this.clipDefinition.setAttribute( 'clipPathUnits', 'userSpaceOnUse' );
              this.block.defs.appendChild( this.clipDefinition ); // TODO: method? evaluate with future usage of defs (not done yet)
              
              this.clipPath = document.createElementNS( scenery.svgns, 'path' );
              this.clipDefinition.appendChild( this.clipPath );
              
              svgGroup.setAttribute( 'clip-path', 'url(#' + clipId + ')' );
            }
            
            this.clipPath.setAttribute( 'd', this.node._clipArea.getSVGPath() );
          } else if ( this.clipDefinition ) {
            svgGroup.removeAttribute( 'clip-path' );
            this.block.defs.removeChild( this.clipDefinition ); // TODO: method? evaluate with future usage of defs (not done yet)
            
            // TODO: consider pooling these?
            this.clipDefinition = null;
            this.clipPath = null;
          }
        }
      }
      
      if ( this.orderDirty ) {
        this.orderDirty = false;
        
        // our instance should have the proper order of children. we check that way.
        var idx = 0;
        var instanceChildren = this.instance.children;
        for ( var i = 0; i < instanceChildren.length; i++ ) {
          var group = instanceChildren[i].lookupSVGGroup( this.block );
          if ( group ) {
            // ensure that the spot in our array (and in the DOM) at [idx] is correct
            if ( this.children[idx] !== group ) {
              // out of order, rearrange
              
              // in the DOM first (since we reference the children array to know what to insertBefore)
              svgGroup.insertBefore( group.svgGroup, idx + 1 >= this.children.length ? null : this.children[idx+1].svgGroup ); // see http://stackoverflow.com/questions/9732624/how-to-swap-dom-child-nodes-in-javascript
              
              // then in our children array
              var oldIndex = _.indexOf( this.children, group );
              assert && assert( oldIndex > idx, 'The item we are moving forward to location [idx] should not have an index less than that' );
              this.children.splice( oldIndex, 1 );
              this.children.splice( idx, 0, group );
            }
            
            // if there was a group for that instance, we move on to the next spot
            idx++;
          }
        }
      }
    },
    
    isReleasable: function() {
      // if we have no parent, we are the rootGroup (the block is responsible for disposing that one)
      return !this.hasSelfDrawable && !this.children.length && this.parent;
    },
    
    dispose: function() {
      if ( this.listeningToTransform ) {
        this.node.offStatic( 'transform', this.transformDirtyListener );
      }
      if ( this.willApplyFilters ) {
        this.node.offStatic( 'opacity', this.opacityDirtyListener );
        this.node.offStatic( 'visible', this.visibilityDirtyListener );
        this.node.offStatic( 'clip', this.clipDirtyListener );
      }
      this.node.offStatic( 'childInserted', this.orderDirtyListener );
      this.node.offStatic( 'childRemoved', this.orderDirtyListener );
      
      this.instance.removeSVGGroup( this );
      
      // clear references
      this.parent = null;
      this.block = null;
      this.instance = null;
      this.node = null;
      this.clipDefinition = null;
      this.clipPath = null;
      cleanArray( this.children );
    }
  };
  
  // @public
  SVGGroup.addDrawable = function( block, drawable ) {
    assert && assert( drawable.instance, 'Instance is required for a drawable to be grouped correctly in SVG' );
    
    var group = SVGGroup.ensureGroupsToInstance( block, drawable.instance );
    group.addSelfDrawable( drawable );
  };
  
  // @public
  SVGGroup.removeDrawable = function( block, drawable ) {
    drawable.instance.lookupSVGGroup( block ).removeSelfDrawable( drawable );
    
    SVGGroup.releaseGroupsToInstance( block, drawable.instance );
  };
  
  // @private
  SVGGroup.ensureGroupsToInstance = function( block, instance ) {
    // TODO: assertions here
    
    var group = instance.lookupSVGGroup( block );
    
    if ( !group ) {
      assert && assert( instance !== block.rootGroup.instance, 'Making sure we do not walk past our rootGroup' );
      
      var parentGroup = SVGGroup.ensureGroupsToInstance( block, instance.parent );
      
      group = SVGGroup.createFromPool( block, instance, parentGroup );
      parentGroup.addChildGroup( group );
    }
    
    return group;
  };
  
  // @private
  SVGGroup.releaseGroupsToInstance = function( block, instance ) {
    var group = instance.lookupSVGGroup( block );
    
    if ( group.isReleasable() ) {
      group.parent.removeChildGroup( group );
      
      SVGGroup.releaseGroupsToInstance( block, instance.parent );
      
      group.freeToPool();
    }
  };
  
  /* jshint -W064 */
  Poolable( SVGGroup, {
    constructorDuplicateFactory: function( pool ) {
      return function( block, instance, parent ) {
        if ( pool.length ) {
          return pool.pop().initialize( block, instance, parent );
        } else {
          return new SVGGroup( block, instance, parent );
        }
      };
    }
  } );
  
  return SVGGroup;
} );
