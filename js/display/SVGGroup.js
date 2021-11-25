// Copyright 2014-2021, University of Colorado Boulder

/**
 * Poolable wrapper for SVG <group> elements. We store state and add listeners directly to the corresponding Node,
 * so that we can set dirty flags and smartly update only things that have changed. This takes a load off of SVGBlock.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import cleanArray from '../../../phet-core/js/cleanArray.js';
import Poolable from '../../../phet-core/js/Poolable.js';
import { scenery, svgns } from '../imports.js';

let globalId = 1;

class SVGGroup {
  /**
   * @mixes Poolable
   *
   * @param {SVGBlock} block
   * @param {Block} instance
   * @param {SVGGroup|null} parent
   */
  constructor( block, instance, parent ) {
    // @public {string}
    this.id = `group${globalId++}`;

    this.initialize( block, instance, parent );
  }

  /**
   * @public
   *
   * @param {SVGBlock} block
   * @param {Block} instance
   * @param {SVGGroup|null} parent
   */
  initialize( block, instance, parent ) {
    //OHTWO TODO: add collapsing groups! they can't have self drawables, transforms, filters, etc., and we probably shouldn't de-collapse groups

    sceneryLog && sceneryLog.SVGGroup && sceneryLog.SVGGroup( `initializing ${this.toString()}` );

    // @public {SVGBlock|null} - Set to null when we're disposing, checked by other code.
    this.block = block;

    // @public {Instance|null} - Set to null when we're disposed.
    this.instance = instance;

    // @public {Node|null} - Set to null when we're disposed
    this.node = instance.trail.lastNode();

    // @public {SVGGroup|null}
    this.parent = parent;

    // @public {Array.<SVGGroup>}
    this.children = cleanArray( this.children );

    // @private {boolean}
    this.hasSelfDrawable = false;

    // @private {SVGSelfDrawable|null}
    this.selfDrawable = null;

    // @private {boolean} - general dirty flag (triggered on any other dirty event)
    this.dirty = true;

    // @private {boolean} - we won't listen for transform changes (or even want to set a transform) if our node is
    // beneath a transform root
    this.willApplyTransforms = this.block.transformRootInstance.trail.nodes.length < this.instance.trail.nodes.length;

    // @private {boolean} - we won't listen for filter changes (or set filters, like opacity or visibility) if our node
    // is beneath a filter root
    this.willApplyFilters = this.block.filterRootInstance.trail.nodes.length < this.instance.trail.nodes.length;

    // transform handling
    this.transformDirty = true;
    this.hasTransform = this.hasTransform !== undefined ? this.hasTransform : false; // persists across disposal
    this.transformDirtyListener = this.transformDirtyListener || this.markTransformDirty.bind( this );
    if ( this.willApplyTransforms ) {
      this.node.transformEmitter.addListener( this.transformDirtyListener );
    }

    // @private {boolean}
    this.filterDirty = true;
    this.visibilityDirty = true;
    this.clipDirty = true;

    // @private {SVGFilterElement|null} - lazily created
    this.filterElement = this.filterElement || null;

    // @private {boolean} - Whether we have an opacity attribute set on our SVG element (persists across disposal)
    this.hasOpacity = this.hasOpacity !== undefined ? this.hasOpacity : false;

    // @private {boolean} - Whether we have a filter element connected to our block (and that is being used with a filter
    // attribute). Since this needs to be cleaned up when we are disposed, this will be set to false when disposed
    // (with the associated attribute and defs reference cleaned up).
    this.hasFilter = false;

    this.clipDefinition = this.clipDefinition !== undefined ? this.clipDefinition : null; // persists across disposal
    this.clipPath = this.clipPath !== undefined ? this.clipPath : null; // persists across disposal
    this.filterChangeListener = this.filterChangeListener || this.onFilterChange.bind( this );
    this.visibilityDirtyListener = this.visibilityDirtyListener || this.onVisibleChange.bind( this );
    this.clipDirtyListener = this.clipDirtyListener || this.onClipChange.bind( this );
    this.node.visibleProperty.lazyLink( this.visibilityDirtyListener );
    if ( this.willApplyFilters ) {
      this.node.filterChangeEmitter.addListener( this.filterChangeListener );
    }
    //OHTWO TODO: remove clip workaround
    this.node.clipAreaProperty.lazyLink( this.clipDirtyListener );

    // for tracking the order of child groups, we use a flag and update (reorder) once per updateDisplay if necessary.
    this.orderDirty = true;
    this.orderDirtyListener = this.orderDirtyListener || this.markOrderDirty.bind( this );
    this.node.childrenChangedEmitter.addListener( this.orderDirtyListener );

    if ( !this.svgGroup ) {
      this.svgGroup = document.createElementNS( svgns, 'g' );
    }

    this.instance.addSVGGroup( this );

    this.block.markDirtyGroup( this ); // so we are marked and updated properly
  }

  /**
   * @private
   *
   * @param {SelfDrawable} drawable
   */
  addSelfDrawable( drawable ) {
    this.selfDrawable = drawable;
    this.svgGroup.insertBefore( drawable.svgElement, this.children.length ? this.children[ 0 ].svgGroup : null );
    this.hasSelfDrawable = true;
  }

  /**
   * @private
   *
   * @param {SelfDrawable} drawable
   */
  removeSelfDrawable( drawable ) {
    this.hasSelfDrawable = false;
    this.svgGroup.removeChild( drawable.svgElement );
    this.selfDrawable = null;
  }

  /**
   * @private
   *
   * @param {SVGGroup} group
   */
  addChildGroup( group ) {
    this.markOrderDirty();

    group.parent = this;
    this.children.push( group );
    this.svgGroup.appendChild( group.svgGroup );
  }

  /**
   * @private
   *
   * @param {SVGGroup} group
   */
  removeChildGroup( group ) {
    this.markOrderDirty();

    group.parent = null;
    this.children.splice( _.indexOf( this.children, group ), 1 );
    this.svgGroup.removeChild( group.svgGroup );
  }

  /**
   * @public
   */
  markDirty() {
    if ( !this.dirty ) {
      this.dirty = true;

      this.block.markDirtyGroup( this );
    }
  }

  /**
   * @public
   */
  markOrderDirty() {
    if ( !this.orderDirty ) {
      this.orderDirty = true;
      this.markDirty();
    }
  }

  /**
   * @public
   */
  markTransformDirty() {
    if ( !this.transformDirty ) {
      this.transformDirty = true;
      this.markDirty();
    }
  }

  /**
   * @private
   */
  onFilterChange() {
    if ( !this.filterDirty ) {
      this.filterDirty = true;
      this.markDirty();
    }
  }

  /**
   * @private
   */
  onVisibleChange() {
    if ( !this.visibilityDirty ) {
      this.visibilityDirty = true;
      this.markDirty();
    }
  }

  /**
   * @private
   */
  onClipChange() {
    if ( !this.clipDirty ) {
      this.clipDirty = true;
      this.markDirty();
    }
  }

  /**
   * @public
   */
  update() {
    sceneryLog && sceneryLog.SVGGroup && sceneryLog.SVGGroup( `update: ${this.toString()}` );

    // we may have been disposed since being marked dirty on our block. we won't have a reference if we are disposed
    if ( !this.block ) {
      return;
    }

    sceneryLog && sceneryLog.SVGGroup && sceneryLog.push();

    const svgGroup = this.svgGroup;

    this.dirty = false;

    if ( this.transformDirty ) {
      this.transformDirty = false;

      sceneryLog && sceneryLog.SVGGroup && sceneryLog.SVGGroup( `transform update: ${this.toString()}` );

      if ( this.willApplyTransforms ) {

        const isIdentity = this.node.transform.isIdentity();

        if ( !isIdentity ) {
          this.hasTransform = true;
          svgGroup.setAttribute( 'transform', this.node.transform.getMatrix().getSVGTransform() );
        }
        else if ( this.hasTransform ) {
          this.hasTransform = false;
          svgGroup.removeAttribute( 'transform' );
        }
      }
      else {
        // we want no transforms if we won't be applying transforms
        if ( this.hasTransform ) {
          this.hasTransform = false;
          svgGroup.removeAttribute( 'transform' );
        }
      }
    }

    if ( this.visibilityDirty ) {
      this.visibilityDirty = false;

      sceneryLog && sceneryLog.SVGGroup && sceneryLog.SVGGroup( `visibility update: ${this.toString()}` );

      svgGroup.style.display = this.node.isVisible() ? '' : 'none';
    }

    // TODO: Check if we can leave opacity separate. If it gets applied "after" then we can have them separate
    if ( this.filterDirty ) {
      this.filterDirty = false;

      sceneryLog && sceneryLog.SVGGroup && sceneryLog.SVGGroup( `filter update: ${this.toString()}` );

      const opacity = this.node.effectiveOpacity;
      if ( this.willApplyFilters && opacity !== 1 ) {
        this.hasOpacity = true;
        svgGroup.setAttribute( 'opacity', opacity );
      }
      else if ( this.hasOpacity ) {
        this.hasOpacity = false;
        svgGroup.removeAttribute( 'opacity' );
      }

      const needsFilter = this.willApplyFilters && this.node._filters.length;
      const filterId = `filter-${this.id}`;

      if ( needsFilter ) {
        // Lazy creation of the filter element (if we haven't already)
        if ( !this.filterElement ) {
          this.filterElement = document.createElementNS( svgns, 'filter' );
          this.filterElement.setAttribute( 'id', filterId );
        }

        // Remove all children of the filter element if we're applying filters (if not, we won't have it attached)
        while ( this.filterElement.firstChild ) {
          this.filterElement.removeChild( this.filterElement.lastChild );
        }

        // Fill in elements into our filter
        let filterRegionPercentageIncrease = 50;
        let inName = 'SourceGraphic';
        const length = this.node._filters.length;
        for ( let i = 0; i < length; i++ ) {
          const filter = this.node._filters[ i ];

          const resultName = i === length - 1 ? undefined : `e${i}`; // Last result should be undefined
          filter.applySVGFilter( this.filterElement, inName, resultName );
          filterRegionPercentageIncrease += filter.filterRegionPercentageIncrease;
          inName = resultName;
        }

        // Bleh, no good way to handle the filter region? https://drafts.fxtf.org/filter-effects/#filter-region
        // If we WANT to track things by their actual display size AND pad pixels, AND copy tons of things... we could
        // potentially use the userSpaceOnUse and pad the proper number of pixels. That sounds like an absolute pain, AND
        // a performance drain and abstraction break.
        const min = `-${toSVGNumber( filterRegionPercentageIncrease )}%`;
        const size = `${toSVGNumber( 2 * filterRegionPercentageIncrease + 100 )}%`;
        this.filterElement.setAttribute( 'x', min );
        this.filterElement.setAttribute( 'y', min );
        this.filterElement.setAttribute( 'width', size );
        this.filterElement.setAttribute( 'height', size );
      }

      if ( needsFilter ) {
        if ( !this.hasFilter ) {
          this.block.defs.appendChild( this.filterElement );
        }
        svgGroup.setAttribute( 'filter', `url(#${filterId})` );
        this.hasFilter = true;
      }
      if ( this.hasFilter && !needsFilter ) {
        svgGroup.removeAttribute( 'filter' );
        this.hasFilter = false;
        this.block.defs.removeChild( this.filterElement );
      }
    }

    if ( this.clipDirty ) {
      this.clipDirty = false;

      sceneryLog && sceneryLog.SVGGroup && sceneryLog.SVGGroup( `clip update: ${this.toString()}` );

      //OHTWO TODO: remove clip workaround (use this.willApplyFilters)
      if ( this.node.clipArea ) {
        if ( !this.clipDefinition ) {
          const clipId = `clip${this.node.getId()}`;

          this.clipDefinition = document.createElementNS( svgns, 'clipPath' );
          this.clipDefinition.setAttribute( 'id', clipId );
          this.clipDefinition.setAttribute( 'clipPathUnits', 'userSpaceOnUse' );
          this.block.defs.appendChild( this.clipDefinition ); // TODO: method? evaluate with future usage of defs (not done yet)

          this.clipPath = document.createElementNS( svgns, 'path' );
          this.clipDefinition.appendChild( this.clipPath );

          svgGroup.setAttribute( 'clip-path', `url(#${clipId})` );
        }

        this.clipPath.setAttribute( 'd', this.node.clipArea.getSVGPath() );
      }
      else if ( this.clipDefinition ) {
        svgGroup.removeAttribute( 'clip-path' );
        this.block.defs.removeChild( this.clipDefinition ); // TODO: method? evaluate with future usage of defs (not done yet)

        // TODO: consider pooling these?
        this.clipDefinition = null;
        this.clipPath = null;
      }
    }

    if ( this.orderDirty ) {
      this.orderDirty = false;

      sceneryLog && sceneryLog.SVGGroup && sceneryLog.SVGGroup( `order update: ${this.toString()}` );
      sceneryLog && sceneryLog.SVGGroup && sceneryLog.push();

      // our instance should have the proper order of children. we check that way.
      let idx = this.children.length - 1;
      const instanceChildren = this.instance.children;
      // iterate backwards, since DOM's insertBefore makes forward iteration more complicated (no insertAfter)
      for ( let i = instanceChildren.length - 1; i >= 0; i-- ) {
        const group = instanceChildren[ i ].lookupSVGGroup( this.block );
        if ( group ) {
          // ensure that the spot in our array (and in the DOM) at [idx] is correct
          if ( this.children[ idx ] !== group ) {
            // out of order, rearrange
            sceneryLog && sceneryLog.SVGGroup && sceneryLog.SVGGroup( `group out of order: ${idx} for ${group.toString()}` );

            // in the DOM first (since we reference the children array to know what to insertBefore)
            // see http://stackoverflow.com/questions/9732624/how-to-swap-dom-child-nodes-in-javascript
            svgGroup.insertBefore( group.svgGroup, idx + 1 >= this.children.length ? null : this.children[ idx + 1 ].svgGroup );

            // then in our children array
            const oldIndex = _.indexOf( this.children, group );
            assert && assert( oldIndex < idx, 'The item we are moving backwards to location [idx] should not have an index greater than that' );
            this.children.splice( oldIndex, 1 );
            this.children.splice( idx, 0, group );
          }
          else {
            sceneryLog && sceneryLog.SVGGroup && sceneryLog.SVGGroup( `group in place: ${idx} for ${group.toString()}` );
          }

          // if there was a group for that instance, we move on to the next spot
          idx--;
        }
      }

      sceneryLog && sceneryLog.SVGGroup && sceneryLog.pop();
    }

    sceneryLog && sceneryLog.SVGGroup && sceneryLog.pop();
  }

  /**
   * @private
   *
   * @returns {boolean}
   */
  isReleasable() {
    // if we have no parent, we are the rootGroup (the block is responsible for disposing that one)
    return !this.hasSelfDrawable && !this.children.length && this.parent;
  }

  /**
   * Releases references
   * @public
   */
  dispose() {
    sceneryLog && sceneryLog.SVGGroup && sceneryLog.SVGGroup( `dispose ${this.toString()}` );
    sceneryLog && sceneryLog.SVGGroup && sceneryLog.push();

    assert && assert( this.children.length === 0, 'Should be empty by now' );

    if ( this.hasFilter ) {
      this.svgGroup.removeAttribute( 'filter' );
      this.hasFilter = false;
      this.block.defs.removeChild( this.filterElement );
    }

    if ( this.willApplyTransforms ) {
      this.node.transformEmitter.removeListener( this.transformDirtyListener );
    }
    this.node.visibleProperty.unlink( this.visibilityDirtyListener );
    if ( this.willApplyFilters ) {
      this.node.filterChangeEmitter.removeListener( this.filterChangeListener );
    }
    //OHTWO TODO: remove clip workaround
    this.node.clipAreaProperty.unlink( this.clipDirtyListener );

    this.node.childrenChangedEmitter.removeListener( this.orderDirtyListener );

    // if our Instance has been disposed, it has already had the reference removed
    if ( this.instance.active ) {
      this.instance.removeSVGGroup( this );
    }

    // remove clipping, since it is defs-based (and we want to keep our defs block clean - could be another layer!)
    if ( this.clipDefinition ) {
      this.svgGroup.removeAttribute( 'clip-path' );
      this.block.defs.removeChild( this.clipDefinition );
      this.clipDefinition = null;
      this.clipPath = null;
    }

    // clear references
    this.parent = null;
    this.block = null;
    this.instance = null;
    this.node = null;
    cleanArray( this.children );
    this.selfDrawable = null;

    // for now
    this.freeToPool();

    sceneryLog && sceneryLog.SVGGroup && sceneryLog.pop();
  }

  /**
   * Returns a string form of this object
   * @public
   *
   * @returns {string}
   */
  toString() {
    return `SVGGroup:${this.block.toString()}_${this.instance.toString()}`;
  }

  /**
   * @public
   *
   * @param {SVGBlock} block
   * @param {Drawable} drawable
   */
  static addDrawable( block, drawable ) {
    assert && assert( drawable.instance, 'Instance is required for a drawable to be grouped correctly in SVG' );

    const group = SVGGroup.ensureGroupsToInstance( block, drawable.instance );
    group.addSelfDrawable( drawable );
  }

  /**
   * @public
   *
   * @param {SVGBlock} block
   * @param {Drawable} drawable
   */
  static removeDrawable( block, drawable ) {
    drawable.instance.lookupSVGGroup( block ).removeSelfDrawable( drawable );

    SVGGroup.releaseGroupsToInstance( block, drawable.instance );
  }

  /**
   * @private
   *
   * @param {SVGBlock} block
   * @param {Instance} instance
   * @returns {SVGGroup}
   */
  static ensureGroupsToInstance( block, instance ) {
    // TODO: assertions here

    let group = instance.lookupSVGGroup( block );

    if ( !group ) {
      assert && assert( instance !== block.rootGroup.instance, 'Making sure we do not walk past our rootGroup' );

      const parentGroup = SVGGroup.ensureGroupsToInstance( block, instance.parent );

      group = SVGGroup.createFromPool( block, instance, parentGroup );
      parentGroup.addChildGroup( group );
    }

    return group;
  }

  /**
   * @private
   *
   * @param {SVGBlock} block
   * @param {Instance} instance
   */
  static releaseGroupsToInstance( block, instance ) {
    const group = instance.lookupSVGGroup( block );

    if ( group.isReleasable() ) {
      const parentGroup = group.parent;
      parentGroup.removeChildGroup( group );

      SVGGroup.releaseGroupsToInstance( block, parentGroup.instance );

      group.dispose();
    }
  }
}

scenery.register( 'SVGGroup', SVGGroup );

Poolable.mixInto( SVGGroup );

export default SVGGroup;