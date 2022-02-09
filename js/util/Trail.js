// Copyright 2013-2021, University of Colorado Boulder

/**
 * Represents a trail (path in the graph) from a 'root' node down to a descendant node.
 * In a DAG, or with different views, there can be more than one trail up from a node,
 * even to the same root node!
 *
 * It has an array of nodes, in order from the 'root' down to the last node,
 * a length, and an array of indices such that node_i.children[index_i] === node_{i+1}.
 *
 * The indices can sometimes become stale when nodes are added and removed, so Trails
 * can have their indices updated with reindex(). It's designed to be as fast as possible
 * on Trails that are already indexed accurately.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../dot/js/Matrix3.js';
import Transform3 from '../../../dot/js/Transform3.js';
import { scenery, Node, TrailPointer } from '../imports.js';

// constants
const ID_SEPARATOR = '-';

class Trail {
  /**
   * @param {Trail|Array.<Node>|Node} [nodes]
   */
  constructor( nodes ) {
    /*
     * Controls the immutability of the trail.
     * If set to true, add/remove descendant/ancestor should fail if assertions are enabled
     * Use setImmutable() or setMutable() to signal a specific type of protection, so it cannot be changed later
     */
    if ( assert ) {
      // @private {boolean|undefined} only do this if assertions are enabled, otherwise we won't access it at all
      this.immutable = undefined;
    }

    if ( nodes instanceof Trail ) {
      // copy constructor (takes advantage of already built index information)
      const otherTrail = nodes;

      this.nodes = otherTrail.nodes.slice( 0 );
      this.length = otherTrail.length;
      this.uniqueId = otherTrail.uniqueId;
      this.indices = otherTrail.indices.slice( 0 );
      return;
    }

    // @public {Array.<Node>} - The main nodes of the trail, in order from root to leaf
    this.nodes = [];

    // @public {number} - Shortcut for the length of nodes.
    this.length = 0;

    // @public {string} - A unique identifier that should only be shared by other trails that are identical to this one.
    this.uniqueId = '';

    // @public {Array.<number>} - indices[x] stores the index of nodes[x] in nodes[x-1]'s children, e.g.
    // nodes[i].children[ indices[i] ] === nodes[i+1]
    this.indices = [];

    if ( nodes ) {
      if ( nodes instanceof Node ) {
        const node = nodes;

        // add just a single node in
        this.addDescendant( node );
      }
      else {
        // process it as an array
        const len = nodes.length;
        for ( let i = 0; i < len; i++ ) {
          this.addDescendant( nodes[ i ] );
        }
      }
    }
  }

  /**
   * Returns a copy of this Trail that can be modified independently
   * @public
   *
   * @returns {Trail}
   */
  copy() {
    return new Trail( this );
  }

  /**
   * Whether the leaf-most Node in our trail will render something
   * @public (scenery-internal)
   *
   * @returns {boolean}
   */
  isPainted() {
    return this.lastNode().isPainted();
  }

  /**
   * Whether all nodes in the trail are still connected from the trail's root to its leaf.
   * @public
   *
   * @returns {boolean}
   */
  isValid() {
    this.reindex();

    const indexLength = this.indices.length;
    for ( let i = 0; i < indexLength; i++ ) {
      if ( this.indices[ i ] < 0 ) {
        return false;
      }
    }

    return true;
  }

  /**
   * This trail is visible only if all nodes on it are marked as visible
   * @public
   *
   * @returns {boolean}
   */
  isVisible() {
    let i = this.nodes.length;
    while ( i-- ) {
      if ( !this.nodes[ i ].isVisible() ) {
        return false;
      }
    }
    return true;
  }

  /**
   * This trail is pdomVisible only if all nodes on it are marked as pdomVisible
   * @public
   *
   * @returns {boolean}
   */
  isPDOMVisible() {
    let i = this.nodes.length;
    while ( i-- ) {
      if ( !this.nodes[ i ].isVisible() || !this.nodes[ i ].isPDOMVisible() ) {
        return false;
      }
    }
    return true;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  getOpacity() {
    let opacity = 1;
    let i = this.nodes.length;
    while ( i-- ) {
      opacity *= this.nodes[ i ].getOpacity();
    }
    return opacity;
  }

  /**
   * Essentially whether this node is visited in the hit-testing operation
   * @public
   *
   * @returns {boolean}
   */
  isPickable() {
    // it won't be if it or any ancestor is pickable: false, or is invisible
    if ( _.some( this.nodes, node => node.pickable === false || node.visible === false ) ) { return false; }

    // if there is any listener or pickable: true, it will be pickable
    if ( _.some( this.nodes, node => node._inputListeners.length > 0 || node.pickableProperty.value === true ) ) { return true; }

    // TODO: Is this even necessary?
    if ( this.lastNode()._picker._subtreePickableCount > 0 ) {
      return true;
    }

    // no listeners or pickable: true, so it will be pruned
    return false;
  }

  /**
   * @public
   *
   * @param {number} index
   * @returns {Node}
   */
  get( index ) {
    if ( index >= 0 ) {
      return this.nodes[ index ];
    }
    else {
      // negative index goes from the end of the array
      return this.nodes[ this.nodes.length + index ];
    }
  }

  /**
   * @public
   *
   * @param {number} startIndex
   * @param {number} endIndex
   * @returns {Trail}
   */
  slice( startIndex, endIndex ) {
    return new Trail( this.nodes.slice( startIndex, endIndex ) );
  }

  /**
   * @public
   *
   * TODO: consider renaming to subtrailToExcluding and subtrailToIncluding?
   *
   * @param {Node} node
   * @param {boolean} [excludeNode]
   * @returns {Trail}
   */
  subtrailTo( node, excludeNode = false ) {
    return this.slice( 0, _.indexOf( this.nodes, node ) + ( excludeNode ? 0 : 1 ) );
  }

  /**
   * @public
   *
   * @returns {boolean}
   */
  isEmpty() {
    return this.nodes.length === 0;
  }

  /**
   * From local to global
   * @public
   *
   * @returns {Matrix3}
   */
  getMatrix() {
    // TODO: performance: can we cache this ever? would need the rootNode to not really change in between
    // this matrix will be modified in place, so always start fresh
    const matrix = Matrix3.identity();

    // from the root up
    const nodes = this.nodes;
    const length = nodes.length;
    for ( let i = 0; i < length; i++ ) {
      matrix.multiplyMatrix( nodes[ i ].getMatrix() );
    }
    return matrix;
  }

  /**
   * From local to next-to-global (ignores root node matrix)
   * @public
   *
   * @returns {Matrix3}
   */
  getAncestorMatrix() {
    // TODO: performance: can we cache this ever? would need the rootNode to not really change in between
    // this matrix will be modified in place, so always start fresh
    const matrix = Matrix3.identity();

    // from the root up
    const nodes = this.nodes;
    const length = nodes.length;
    for ( let i = 1; i < length; i++ ) {
      matrix.multiplyMatrix( nodes[ i ].getMatrix() );
    }
    return matrix;
  }

  /**
   * From parent to global
   * @public
   *
   * @returns {Matrix3}
   */
  getParentMatrix() {
    // this matrix will be modified in place, so always start fresh
    const matrix = Matrix3.identity();

    // from the root up
    const nodes = this.nodes;
    const length = nodes.length;
    for ( let i = 0; i < length - 1; i++ ) {
      matrix.multiplyMatrix( nodes[ i ].getMatrix() );
    }
    return matrix;
  }

  /**
   * From local to global
   * @public
   *
   * @returns {Transform3}
   */
  getTransform() {
    return new Transform3( this.getMatrix() );
  }

  /**
   * From parent to global
   * @public
   *
   * @returns {Transform3}
   */
  getParentTransform() {
    return new Transform3( this.getParentMatrix() );
  }

  /**
   * @public
   *
   * @param {Node} node
   * @param {number} [index]
   * @returns {Trail} - For chaining
   */
  addAncestor( node, index ) {
    assert && assert( !this.immutable, 'cannot modify an immutable Trail with addAncestor' );
    assert && assert( node, 'cannot add falsy value to a Trail' );


    if ( this.nodes.length ) {
      const oldRoot = this.nodes[ 0 ];
      this.indices.unshift( index === undefined ? _.indexOf( node._children, oldRoot ) : index );
    }
    this.nodes.unshift( node );

    this.length++;
    // accelerated version of this.updateUniqueId()
    this.uniqueId = ( this.uniqueId ? node.id + ID_SEPARATOR + this.uniqueId : `${node.id}` );

    return this;
  }

  /**
   * @public
   *
   * @returns {Trail} - For chaining
   */
  removeAncestor() {
    assert && assert( !this.immutable, 'cannot modify an immutable Trail with removeAncestor' );
    assert && assert( this.length > 0, 'cannot remove a Node from an empty trail' );

    this.nodes.shift();
    if ( this.indices.length ) {
      this.indices.shift();
    }

    this.length--;
    this.updateUniqueId();

    return this;
  }

  /**
   * @public
   *
   * @param {Node} node
   * @param {number} index
   * @returns {Trail} - For chaining
   */
  addDescendant( node, index ) {
    assert && assert( !this.immutable, 'cannot modify an immutable Trail with addDescendant' );
    assert && assert( node, 'cannot add falsy value to a Trail' );


    if ( this.nodes.length ) {
      const parent = this.lastNode();
      this.indices.push( index === undefined ? _.indexOf( parent._children, node ) : index );
    }
    this.nodes.push( node );

    this.length++;
    // accelerated version of this.updateUniqueId()
    this.uniqueId = ( this.uniqueId ? this.uniqueId + ID_SEPARATOR + node.id : `${node.id}` );

    return this;
  }

  /**
   * @public
   *
   * @returns {Trail} - For chaining
   */
  removeDescendant() {
    assert && assert( !this.immutable, 'cannot modify an immutable Trail with removeDescendant' );
    assert && assert( this.length > 0, 'cannot remove a Node from an empty trail' );

    this.nodes.pop();
    if ( this.indices.length ) {
      this.indices.pop();
    }

    this.length--;
    this.updateUniqueId();

    return this;
  }

  /**
   * @public
   *
   * @param {Trail} trail
   */
  addDescendantTrail( trail ) {
    const length = trail.length;
    if ( length ) {
      this.addDescendant( trail.nodes[ 0 ] );
    }
    for ( let i = 1; i < length; i++ ) {
      this.addDescendant( trail.nodes[ i ], this.indices[ i - 1 ] );
    }
  }

  /**
   * @public
   *
   * @param {Trail} trail
   */
  removeDescendantTrail( trail ) {
    const length = trail.length;
    for ( let i = length - 1; i >= 0; i-- ) {
      assert && assert( this.lastNode() === trail.nodes[ i ] );

      this.removeDescendant();
    }
  }

  /**
   * Refreshes the internal index references (important if any children arrays were modified!)
   * @public
   */
  reindex() {
    const length = this.length;
    for ( let i = 1; i < length; i++ ) {
      // only replace indices where they have changed (this was a performance hotspot)
      const currentIndex = this.indices[ i - 1 ];
      const baseNode = this.nodes[ i - 1 ];

      if ( baseNode._children[ currentIndex ] !== this.nodes[ i ] ) {
        this.indices[ i - 1 ] = _.indexOf( baseNode._children, this.nodes[ i ] );
      }
    }
  }

  /**
   * @public
   *
   * @returns {Trail} - For chaining
   */
  setImmutable() {
    // if assertions are disabled, we hope this is inlined as a no-op
    if ( assert ) {
      assert( this.immutable !== false, 'A trail cannot be made immutable after being flagged as mutable' );
      this.immutable = true;
    }

    // TODO: consider setting mutators to null here instead of the function call check (for performance, and profile the differences)

    return this; // allow chaining
  }

  /**
   * @public
   *
   * @returns {Trail} - For chaining
   */
  setMutable() {
    // if assertions are disabled, we hope this is inlined as a no-op
    if ( assert ) {
      assert( this.immutable !== true, 'A trail cannot be made mutable after being flagged as immutable' );
      this.immutable = false;
    }

    return this; // allow chaining
  }

  /**
   * @public
   *
   * @returns {boolean}
   */
  areIndicesValid() {
    for ( let i = 1; i < this.length; i++ ) {
      const currentIndex = this.indices[ i - 1 ];
      if ( this.nodes[ i - 1 ]._children[ currentIndex ] !== this.nodes[ i ] ) {
        return false;
      }
    }
    return true;
  }

  /**
   * @public
   *
   * @param {Trail} other
   * @returns {boolean}
   */
  equals( other ) {
    if ( this.length !== other.length ) {
      return false;
    }

    for ( let i = 0; i < this.nodes.length; i++ ) {
      if ( this.nodes[ i ] !== other.nodes[ i ] ) {
        return false;
      }
    }

    return true;
  }

  /**
   * Returns a new Trail from the root up to the parameter node.
   * @public
   *
   * @param {Node} node
   * @returns {Trail}
   */
  upToNode( node ) {
    const nodeIndex = _.indexOf( this.nodes, node );
    assert && assert( nodeIndex >= 0, 'Trail does not contain the node' );
    return this.slice( 0, _.indexOf( this.nodes, node ) + 1 );
  }

  /**
   * Whether this trail contains the complete 'other' trail, but with added descendants afterwards.
   * @public
   *
   * @param {Trail} other - is other a subset of this trail?
   * @param {boolean} allowSameTrail
   * @returns {boolean}
   */
  isExtensionOf( other, allowSameTrail ) {
    if ( this.length <= other.length - ( allowSameTrail ? 1 : 0 ) ) {
      return false;
    }

    for ( let i = 0; i < other.nodes.length; i++ ) {
      if ( this.nodes[ i ] !== other.nodes[ i ] ) {
        return false;
      }
    }

    return true;
  }

  /**
   * Returns whether a given node is contained in the trail.
   * @public
   *
   * @param {Node} node
   * @returns {boolean}
   */
  containsNode( node ) {
    return _.includes( this.nodes, node );
  }

  /**
   * A transform from our local coordinate frame to the other trail's local coordinate frame
   * @public
   *
   * @param {Trail} otherTrail
   * @returns {Transform3}
   */
  getTransformTo( otherTrail ) {
    return new Transform3( this.getMatrixTo( otherTrail ) );
  }

  /**
   * Returns a matrix that transforms a point in our last node's local coordinate frame to the other trail's last node's
   * local coordinate frame
   * @public
   *
   * @param {Trail} otherTrail
   * @returns {Matrix3}
   */
  getMatrixTo( otherTrail ) {
    this.reindex();
    otherTrail.reindex();

    const branchIndex = this.getBranchIndexTo( otherTrail );
    let idx;

    let matrix = Matrix3.IDENTITY;

    // walk our transform down, prepending
    for ( idx = this.length - 1; idx >= branchIndex; idx-- ) {
      matrix = this.nodes[ idx ].getMatrix().timesMatrix( matrix );
    }

    // walk our transform up, prepending inverses
    for ( idx = branchIndex; idx < otherTrail.length; idx++ ) {
      matrix = otherTrail.nodes[ idx ].getTransform().getInverse().timesMatrix( matrix );
    }

    return matrix;
  }

  /**
   * Returns the first index that is different between this trail and the other trail.
   * @public
   *
   * If the trails are identical, the index should be equal to the trail's length.
   *
   * @param {Trail} otherTrail
   * @returns {number}
   */
  getBranchIndexTo( otherTrail ) {
    assert && assert( this.nodes[ 0 ] === otherTrail.nodes[ 0 ], 'To get a branch index, the trails must have the same root' );

    let branchIndex;

    const min = Math.min( this.length, otherTrail.length );
    for ( branchIndex = 0; branchIndex < min; branchIndex++ ) {
      if ( this.nodes[ branchIndex ] !== otherTrail.nodes[ branchIndex ] ) {
        break;
      }
    }

    return branchIndex;
  }

  /**
   * Returns the last (largest) index into the trail's nodes that has inputEnabled=true.
   * @public
   *
   * @returns {number}
   */
  getLastInputEnabledIndex() {
    // Determine how far up the Trail input is determined. The first node with !inputEnabled and after will not have
    // events fired (see https://github.com/phetsims/sun/issues/257)
    let trailStartIndex = -1;
    for ( let j = 0; j < this.length; j++ ) {
      if ( !this.nodes[ j ].inputEnabled ) {
        break;
      }

      trailStartIndex = j;
    }

    return trailStartIndex;
  }

  /**
   * Returns the leaf-most index, unless there is a Node with inputEnabled=false (in which case, the lowest index
   * for those matching Nodes are returned).
   * @public
   *
   * @returns {number}
   */
  getCursorCheckIndex() {
    return this.getLastInputEnabledIndex();
  }

  /**
   * @public
   *
   * TODO: phase out in favor of get()
   *
   * @param {number} offset
   * @returns {Node}
   */
  nodeFromTop( offset ) {
    return this.nodes[ this.length - 1 - offset ];
  }

  /**
   * @public
   *
   * @returns {Node}
   */
  lastNode() {
    return this.nodeFromTop( 0 );
  }

  /**
   * @public
   *
   * @returns {Node}
   */
  rootNode() {
    return this.nodes[ 0 ];
  }

  /**
   * Returns the previous graph trail in the order of self-rendering
   * @public
   *
   * @returns {Trail}
   */
  previous() {
    if ( this.nodes.length <= 1 ) {
      return null;
    }

    const top = this.nodeFromTop( 0 );
    const parent = this.nodeFromTop( 1 );

    const parentIndex = _.indexOf( parent._children, top );
    assert && assert( parentIndex !== -1 );
    const arr = this.nodes.slice( 0, this.nodes.length - 1 );
    if ( parentIndex === 0 ) {
      // we were the first child, so give it the trail to the parent
      return new Trail( arr );
    }
    else {
      // previous child
      arr.push( parent._children[ parentIndex - 1 ] );

      // and find its last terminal
      while ( arr[ arr.length - 1 ]._children.length !== 0 ) {
        const last = arr[ arr.length - 1 ];
        arr.push( last._children[ last._children.length - 1 ] );
      }

      return new Trail( arr );
    }
  }

  /**
   * Like previous(), but keeps moving back until the trail goes to a node with isPainted() === true
   * @public
   *
   * @returns {Trail}
   */
  previousPainted() {
    let result = this.previous();
    while ( result && !result.isPainted() ) {
      result = result.previous();
    }
    return result;
  }

  /**
   * In the order of self-rendering
   * @public
   *
   * @returns {Trail}
   */
  next() {
    const arr = this.nodes.slice( 0 );

    const top = this.nodeFromTop( 0 );
    if ( top._children.length > 0 ) {
      // if we have children, return the first child
      arr.push( top._children[ 0 ] );
      return new Trail( arr );
    }
    else {
      // walk down and attempt to find the next parent
      let depth = this.nodes.length - 1;

      while ( depth > 0 ) {
        const node = this.nodes[ depth ];
        const parent = this.nodes[ depth - 1 ];

        arr.pop(); // take off the node so we can add the next sibling if it exists

        const index = _.indexOf( parent._children, node );
        if ( index !== parent._children.length - 1 ) {
          // there is another (later) sibling. use that!
          arr.push( parent._children[ index + 1 ] );
          return new Trail( arr );
        }
        else {
          depth--;
        }
      }

      // if we didn't reach a later sibling by now, it doesn't exist
      return null;
    }
  }

  /**
   * Like next(), but keeps moving back until the trail goes to a node with isPainted() === true
   * @public
   *
   * @returns {Trail}
   */
  nextPainted() {
    let result = this.next();
    while ( result && !result.isPainted() ) {
      result = result.next();
    }
    return result;
  }

  /**
   * Calls callback( trail ) for this trail, and each descendant trail. If callback returns true, subtree will be skipped
   * @public
   *
   * @param {function(Trail)} callback
   */
  eachTrailUnder( callback ) {
    // TODO: performance: should be optimized to be much faster, since we don't have to deal with the before/after
    new TrailPointer( this, true ).eachTrailBetween( new TrailPointer( this, false ), callback );
  }

  /*
   * Standard Java-style compare. -1 means this trail is before (under) the other trail, 0 means equal, and 1 means this trail is
   * after (on top of) the other trail.
   * A shorter subtrail will compare as -1.
   * @public
   *
   * Assumes that the Trails are properly indexed. If not, please reindex them!
   *
   * Comparison is for the rendering order, so an ancestor is 'before' a descendant
   *
   * @param {Trail} other
   * @returns {boolean}
   */
  compare( other ) {
    assert && assert( !this.isEmpty(), 'cannot compare with an empty trail' );
    assert && assert( !other.isEmpty(), 'cannot compare with an empty trail' );
    assert && assert( this.nodes[ 0 ] === other.nodes[ 0 ], 'for Trail comparison, trails must have the same root node' );
    assertSlow && assertSlow( this.areIndicesValid(), `Trail.compare this.areIndicesValid() failed on ${this.toString()}` );
    assertSlow && assertSlow( other.areIndicesValid(), `Trail.compare other.areIndicesValid() failed on ${other.toString()}` );

    const minNodeIndex = Math.min( this.nodes.length, other.nodes.length );
    for ( let i = 0; i < minNodeIndex; i++ ) {
      if ( this.nodes[ i ] !== other.nodes[ i ] ) {
        if ( this.nodes[ i - 1 ].children.indexOf( this.nodes[ i ] ) < other.nodes[ i - 1 ].children.indexOf( other.nodes[ i ] ) ) {
          return -1;
        }
        else {
          return 1;
        }
      }
    }

    // we scanned through and no nodes were different (one is a subtrail of the other)
    if ( this.nodes.length < other.nodes.length ) {
      return -1;
    }
    else if ( this.nodes.length > other.nodes.length ) {
      return 1;
    }
    else {
      return 0;
    }
  }

  /**
   * @public
   *
   * @param {Trail} other
   * @returns {boolean}
   */
  isBefore( other ) {
    return this.compare( other ) === -1;
  }

  /**
   * @public
   *
   * @param {Trail} other
   * @returns {boolean}
   */
  isAfter( other ) {
    return this.compare( other ) === 1;
  }

  /**
   * @public
   *
   * @param {Vector2} point
   * @returns {Vector2}
   */
  localToGlobalPoint( point ) {
    // TODO: performance: multiple timesVector2 calls up the chain is probably faster
    return this.getMatrix().timesVector2( point );
  }

  /**
   * @public
   *
   * @param {Bounds2} bounds
   * @returns {Bounds2}
   */
  localToGlobalBounds( bounds ) {
    return bounds.transformed( this.getMatrix() );
  }

  /**
   * @public
   *
   * @param {Vector2} point
   * @returns {Vector2}
   */
  globalToLocalPoint( point ) {
    return this.getTransform().inversePosition2( point );
  }

  /**
   * @public
   *
   * @param {Bounds2} bounds
   * @returns {Bounds2}
   */
  globalToLocalBounds( bounds ) {
    return this.getTransform().inverseBounds2( bounds );
  }

  /**
   * @public
   *
   * @param {Vector2} point
   * @returns {Vector2}
   */
  parentToGlobalPoint( point ) {
    // TODO: performance: multiple timesVector2 calls up the chain is probably faster
    return this.getParentMatrix().timesVector2( point );
  }

  /**
   * @public
   *
   * @param {Bounds2} bounds
   * @returns {Bounds2}
   */
  parentToGlobalBounds( bounds ) {
    return bounds.transformed( this.getParentMatrix() );
  }

  /**
   * @public
   *
   * @param {Vector2} point
   * @returns {Vector2}
   */
  globalToParentPoint( point ) {
    return this.getParentTransform().inversePosition2( point );
  }

  /**
   * @public
   *
   * @param {Bounds2} bounds
   * @returns {Bounds2}
   */
  globalToParentBounds( bounds ) {
    return this.getParentTransform().inverseBounds2( bounds );
  }

  /**
   * @private
   */
  updateUniqueId() {
    // string concatenation is faster, see http://jsperf.com/string-concat-vs-joins
    let result = '';
    const len = this.nodes.length;
    if ( len > 0 ) {
      result += this.nodes[ 0 ]._id;
    }
    for ( let i = 1; i < len; i++ ) {
      result += ID_SEPARATOR + this.nodes[ i ]._id;
    }
    this.uniqueId = result;
    // this.uniqueId = _.map( this.nodes, function( node ) { return node.getId(); } ).join( '-' );
  }

  /**
   * Concatenates the unique IDs of nodes in the trail, so that we can do id-based lookups
   * @public
   *
   * @returns {string}
   */
  getUniqueId() {
    // sanity checks
    if ( assert ) {
      const oldUniqueId = this.uniqueId;
      this.updateUniqueId();
      assert( oldUniqueId === this.uniqueId );
    }
    return this.uniqueId;
  }

  /**
   * Returns a string form of this object
   * @public
   *
   * @returns {string}
   */
  toString() {
    this.reindex();
    if ( !this.length ) {
      return 'Empty Trail';
    }
    return `[Trail ${this.indices.join( '.' )} ${this.getUniqueId()}]`;
  }

  /**
   * Cleaner string form which will show class names. Not optimized by any means, meant for debugging.
   * @public
   *
   * @returns {string}
   */
  toPathString() {
    return _.map( this.nodes, n => {
      let string = n.constructor.name;
      if ( string === 'Node' ) {
        string = '.';
      }
      return string;
    } ).join( '/' );
  }

  /**
   * Returns a debugging string ideal for logged output.
   * @public
   *
   * @returns {string}
   */
  toDebugString() {
    return `${this.toString()} ${this.toPathString()}`;
  }

  /**
   * Like eachTrailBetween, but only fires for painted trails. If callback returns true, subtree will be skipped
   * @public
   *
   * @param {Trail} a
   * @param {Trail} b
   * @param {function(Trail)} callback
   * @param {boolean} excludeEndTrails
   * @param {Node} rootNode
   */
  static eachPaintedTrailBetween( a, b, callback, excludeEndTrails, rootNode ) {
    Trail.eachTrailBetween( a, b, trail => {
      if ( trail && trail.isPainted() ) {
        return callback( trail );
      }
      return false;
    }, excludeEndTrails, rootNode );
  }

  /**
   * Global way of iterating across trails. when callback returns true, subtree will be skipped
   * @public
   *
   * @param {Trail} a
   * @param {Trail} b
   * @param {function(Trail)} callback
   * @param {boolean} excludeEndTrails
   * @param {Node} rootNode
   */
  static eachTrailBetween( a, b, callback, excludeEndTrails, rootNode ) {
    const aPointer = a ? new TrailPointer( a.copy(), true ) : new TrailPointer( new Trail( rootNode ), true );
    const bPointer = b ? new TrailPointer( b.copy(), true ) : new TrailPointer( new Trail( rootNode ), false );

    // if we are excluding endpoints, just bump the pointers towards each other by one step
    if ( excludeEndTrails ) {
      aPointer.nestedForwards();
      bPointer.nestedBackwards();

      // they were adjacent, so no callbacks will be executed
      if ( aPointer.compareNested( bPointer ) === 1 ) {
        return;
      }
    }

    aPointer.depthFirstUntil( bPointer, pointer => {
      if ( pointer.isBefore ) {
        return callback( pointer.trail );
      }
      return false;
    }, false );
  }

  /**
   * The index at which the two trails diverge. If a.length === b.length === branchIndex, the trails are identical
   * @public
   *
   * @param {Trail} a
   * @param {Trail} b
   * @returns {number}
   */
  static branchIndex( a, b ) {
    assert && assert( a.nodes[ 0 ] === b.nodes[ 0 ], 'Branch changes require roots to be the same' );
    let branchIndex;
    const shortestLength = Math.min( a.length, b.length );
    for ( branchIndex = 0; branchIndex < shortestLength; branchIndex++ ) {
      if ( a.nodes[ branchIndex ] !== b.nodes[ branchIndex ] ) {
        break;
      }
    }
    return branchIndex;
  }

  /**
   * The subtrail from the root that both trails share
   * @public
   *
   * @param {Trail} a
   * @param {Trail} b
   * @returns {Trail}
   */
  static sharedTrail( a, b ) {
    return a.slice( 0, Trail.branchIndex( a, b ) );
  }

  /**
   * @public
   *
   * @param {Array.<Trail>} trailResults - Will be muted by appending matching trails
   * @param {Trail} trail
   * @param {function(Node):boolean} predicate
   */
  static appendAncestorTrailsWithPredicate( trailResults, trail, predicate ) {
    const root = trail.rootNode();

    if ( predicate( root ) ) {
      trailResults.push( trail.copy() );
    }

    const parentCount = root._parents.length;
    for ( let i = 0; i < parentCount; i++ ) {
      const parent = root._parents[ i ];

      trail.addAncestor( parent );
      Trail.appendAncestorTrailsWithPredicate( trailResults, trail, predicate );
      trail.removeAncestor();
    }
  }

  /**
   * @public
   *
   * @param {Array.<Trail>} trailResults - Will be muted by appending matching trails
   * @param {Trail} trail
   * @param {function(Node):boolean} predicate
   */
  static appendDescendantTrailsWithPredicate( trailResults, trail, predicate ) {
    const lastNode = trail.lastNode();

    if ( predicate( lastNode ) ) {
      trailResults.push( trail.copy() );
    }

    const childCount = lastNode._children.length;
    for ( let i = 0; i < childCount; i++ ) {
      const child = lastNode._children[ i ];

      trail.addDescendant( child, i );
      Trail.appendDescendantTrailsWithPredicate( trailResults, trail, predicate );
      trail.removeDescendant();
    }
  }

  /*
   * Fires subtree(trail) or self(trail) on the callbacks to create disjoint subtrees (trails) that cover exactly the nodes
   * inclusively between a and b in rendering order.
   * We try to consolidate these as much as possible.
   * @public
   *
   * "a" and "b" are treated like self painted trails in the rendering order
   *
   *
   * Example tree:
   *   a
   *   - b
   *   --- c
   *   --- d
   *   - e
   *   --- f
   *   ----- g
   *   ----- h
   *   ----- i
   *   --- j
   *   ----- k
   *   - l
   *   - m
   *   --- n
   *
   * spannedSubtrees( a, a ) -> self( a );
   * spannedSubtrees( c, n ) -> subtree( a ); NOTE: if b is painted, that wouldn't work!
   * spannedSubtrees( h, l ) -> subtree( h ); subtree( i ); subtree( j ); self( l );
   * spannedSubtrees( c, i ) -> [b,f] --- wait, include e self?
   *
   * @param {Trail} a
   * @param {Trail} b
   */
  static spannedSubtrees( a, b ) {
    // assert && assert( a.nodes[0] === b.nodes[0], 'Spanned subtrees for a and b requires that a and b have the same root' );

    // a.reindex();
    // b.reindex();

    // var subtrees = [];

    // var branchIndex = Trail.branchIndex( a, b );
    // assert && assert( branchIndex > 0, 'Branch index should always be > 0' );

    // if ( a.length === branchIndex && b.length === branchIndex ) {
    //   // the two trails are equal
    //   subtrees.push( a );
    // } else {
    //   // find the first place where our start isn't the first child
    //   for ( var before = a.length - 1; before >= branchIndex; before-- ) {
    //     if ( a.indices[before-1] !== 0 ) {
    //       break;
    //     }
    //   }

    //   // find the first place where our end isn't the last child
    //   for ( var after = a.length - 1; after >= branchIndex; after-- ) {
    //     if ( b.indices[after-1] !== b.nodes[after-1]._children.length - 1 ) {
    //       break;
    //     }
    //   }

    //   if ( before < branchIndex && after < branchIndex ) {
    //     // we span the entire tree up to nodes[branchIndex-1], so return only that subtree
    //     subtrees.push( a.slice( 0, branchIndex ) );
    //   } else {
    //     // walk the subtrees down from the start
    //     for ( var ia = before; ia >= branchIndex; ia-- ) {
    //       subtrees.push( a.slice( 0, ia + 1 ) );
    //     }

    //     // walk through the middle
    //     var iStart = a.indices[branchIndex-1];
    //     var iEnd = b.indices[branchIndex-1];
    //     var base = a.slice( 0, branchIndex );
    //     var children = base.lastNode()._children;
    //     for ( var im = iStart; im <= iEnd; im++ ) {
    //       subtrees.push( base.copy().addDescendant( children[im], im ) );
    //     }

    //     // walk the subtrees up to the end
    //     for ( var ib = branchIndex; ib <= after; ib++ ) {
    //       subtrees.push( b.slice( 0, ib + 1 ) );
    //     }
    //   }
    // }

    // return subtrees;
  }

  /**
   * Re-create a trail to a root node from an existing Trail id. The rootNode must have the same Id as the first
   * Node id of uniqueId.
   * @public
   *
   * @param {Node} rootNode - the root of the trail being created
   * @param {string} uniqueId - integers separated by ID_SEPARATOR, see getUniqueId
   * @returns {Trail}
   */
  static fromUniqueId( rootNode, uniqueId ) {
    const trailIds = uniqueId.split( ID_SEPARATOR );
    const trailIdNumbers = trailIds.map( id => parseInt( id, 10 ) );

    let currentNode = rootNode;

    const rootId = trailIdNumbers.shift();
    const nodes = [ currentNode ];

    assert && assert( rootId === rootNode.id );

    while ( trailIdNumbers.length > 0 ) {
      const trailId = trailIdNumbers.shift();

      // if accessible order is set, the trail might not match the hierarchy of children - search through nodes
      // in pdomOrder first because pdomOrder is an override for scene graph structure
      const pdomOrder = currentNode.pdomOrder || [];
      const children = pdomOrder.concat( currentNode.children );
      for ( let j = 0; j < children.length; j++ ) {

        // pdomOrder supports null entries to fill in with default order
        if ( children[ j ] !== null && children[ j ].id === trailId ) {
          const childAlongTrail = children[ j ];
          nodes.push( childAlongTrail );
          currentNode = childAlongTrail;

          break;
        }

        assert && assert( j !== children.length - 1, 'unable to find node from unique Trail id' );
      }
    }

    return new Trail( nodes );
  }
}

scenery.register( 'Trail', Trail );
export default Trail;