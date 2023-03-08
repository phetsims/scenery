// Copyright 2013-2023, University of Colorado Boulder

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

import Bounds2 from '../../../dot/js/Bounds2.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import Transform3 from '../../../dot/js/Transform3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import { Node, PDOMUtils, scenery, TrailPointer } from '../imports.js';

// constants
const ID_SEPARATOR = PDOMUtils.PDOM_UNIQUE_ID_SEPARATOR;

export type TrailCallback = ( ( trail: Trail ) => boolean ) | ( ( trail: Trail ) => void );

export default class Trail {

  // The main nodes of the trail, in order from root to leaf
  public nodes: Node[];

  // Shortcut for the length of nodes.
  public length: number;

  // A unique identifier that should only be shared by other trails that are identical to this one.
  public uniqueId: string;

  // indices[x] stores the index of nodes[x] in nodes[x-1]'s children, e.g.
  // nodes[i].children[ indices[i] ] === nodes[i+1]
  public indices: number[];

  // Controls the immutability of the trail.
  // If set to true, add/remove descendant/ancestor should fail if assertions are enabled
  // Use setImmutable() or setMutable() to signal a specific type of protection, so it cannot be changed later
  private immutable?: boolean;

  /**
   * @param [nodes]
   */
  public constructor( nodes?: Trail | Node[] | Node ) {
    if ( assert ) {
      // Only do this if assertions are enabled, otherwise we won't access it at all
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

    this.nodes = [];
    this.length = 0;
    this.uniqueId = '';
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
   */
  public copy(): Trail {
    return new Trail( this );
  }

  /**
   * Whether the leaf-most Node in our trail will render something (scenery-internal)
   */
  public isPainted(): boolean {
    return this.lastNode().isPainted();
  }

  /**
   * Whether all nodes in the trail are still connected from the trail's root to its leaf.
   */
  public isValid(): boolean {
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
   */
  public isVisible(): boolean {
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
   */
  public isPDOMVisible(): boolean {
    let i = this.nodes.length;
    while ( i-- ) {
      if ( !this.nodes[ i ].isVisible() || !this.nodes[ i ].isPDOMVisible() ) {
        return false;
      }
    }
    return true;
  }

  public getOpacity(): number {
    let opacity = 1;
    let i = this.nodes.length;
    while ( i-- ) {
      opacity *= this.nodes[ i ].getOpacity();
    }
    return opacity;
  }

  /**
   * Essentially whether this node is visited in the hit-testing operation
   */
  public isPickable(): boolean {
    // it won't be if it or any ancestor is pickable: false, or is invisible
    if ( _.some( this.nodes, node => node.pickable === false || !node.visible ) ) { return false; }

    // if there is any listener or pickable: true, it will be pickable
    if ( _.some( this.nodes, node => node._inputListeners.length > 0 || node.pickableProperty.value === true ) ) { return true; }

    // no listeners or pickable: true, so it will be pruned
    return false;
  }

  public get( index: number ): Node {
    if ( index >= 0 ) {
      return this.nodes[ index ];
    }
    else {
      // negative index goes from the end of the array
      return this.nodes[ this.nodes.length + index ];
    }
  }

  public slice( startIndex: number, endIndex?: number ): Trail {
    return new Trail( this.nodes.slice( startIndex, endIndex ) );
  }

  /**
   * TODO: consider renaming to subtrailToExcluding and subtrailToIncluding?
   */
  public subtrailTo( node: Node, excludeNode = false ): Trail {
    return this.slice( 0, _.indexOf( this.nodes, node ) + ( excludeNode ? 0 : 1 ) );
  }

  public isEmpty(): boolean {
    return this.nodes.length === 0;
  }

  /**
   * Returns the matrix multiplication of our selected nodes transformation matrices.
   *
   * @param startingIndex - Include nodes matrices starting from this index (inclusive)
   * @param endingIndex - Include nodes matrices up to this index (exclusive)
   */
  public getMatrixConcatenation( startingIndex: number, endingIndex: number ): Matrix3 {
    // TODO: performance: can we cache this ever? would need the rootNode to not really change in between
    // this matrix will be modified in place, so always start fresh
    const matrix = Matrix3.identity();

    // from the root up
    const nodes = this.nodes;
    for ( let i = startingIndex; i < endingIndex; i++ ) {
      matrix.multiplyMatrix( nodes[ i ].getMatrix() );
    }
    return matrix;
  }

  /**
   * From local to global
   *
   * e.g. local coordinate frame of the leaf node to the parent coordinate frame of the root node
   */
  public getMatrix(): Matrix3 {
    return this.getMatrixConcatenation( 0, this.nodes.length );
  }

  /**
   * From local to next-to-global (ignores root node matrix)
   *
   * e.g. local coordinate frame of the leaf node to the local coordinate frame of the root node
   */
  public getAncestorMatrix(): Matrix3 {
    return this.getMatrixConcatenation( 1, this.nodes.length );
  }

  /**
   * From parent to global
   *
   * e.g. parent coordinate frame of the leaf node to the parent coordinate frame of the root node
   */
  public getParentMatrix(): Matrix3 {
    return this.getMatrixConcatenation( 0, this.nodes.length - 1 );
  }

  /**
   * From parent to next-to-global (ignores root node matrix)
   *
   * e.g. parent coordinate frame of the leaf node to the local coordinate frame of the root node
   */
  public getAncestorParentMatrix(): Matrix3 {
    return this.getMatrixConcatenation( 1, this.nodes.length - 1 );
  }

  /**
   * From local to global
   *
   * e.g. local coordinate frame of the leaf node to the parent coordinate frame of the root node
   */
  public getTransform(): Transform3 {
    return new Transform3( this.getMatrix() );
  }

  /**
   * From parent to global
   *
   * e.g. parent coordinate frame of the leaf node to the parent coordinate frame of the root node
   */
  public getParentTransform(): Transform3 {
    return new Transform3( this.getParentMatrix() );
  }

  public addAncestor( node: Node, index?: number ): this {
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

  public removeAncestor(): this {
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

  public addDescendant( node: Node, index?: number ): this {
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

  public removeDescendant(): this {
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

  public addDescendantTrail( trail: Trail ): void {
    const length = trail.length;
    if ( length ) {
      this.addDescendant( trail.nodes[ 0 ] );
    }
    for ( let i = 1; i < length; i++ ) {
      this.addDescendant( trail.nodes[ i ], this.indices[ i - 1 ] );
    }
  }

  public removeDescendantTrail( trail: Trail ): void {
    const length = trail.length;
    for ( let i = length - 1; i >= 0; i-- ) {
      assert && assert( this.lastNode() === trail.nodes[ i ] );

      this.removeDescendant();
    }
  }

  /**
   * Refreshes the internal index references (important if any children arrays were modified!)
   */
  public reindex(): void {
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

  public setImmutable(): this {
    // if assertions are disabled, we hope this is inlined as a no-op
    if ( assert ) {
      assert( this.immutable !== false, 'A trail cannot be made immutable after being flagged as mutable' );
      this.immutable = true;
    }

    // TODO: consider setting mutators to null here instead of the function call check (for performance, and profile the differences)

    return this; // allow chaining
  }

  public setMutable(): this {
    // if assertions are disabled, we hope this is inlined as a no-op
    if ( assert ) {
      assert( this.immutable !== true, 'A trail cannot be made mutable after being flagged as immutable' );
      this.immutable = false;
    }

    return this; // allow chaining
  }

  public areIndicesValid(): boolean {
    for ( let i = 1; i < this.length; i++ ) {
      const currentIndex = this.indices[ i - 1 ];
      if ( this.nodes[ i - 1 ]._children[ currentIndex ] !== this.nodes[ i ] ) {
        return false;
      }
    }
    return true;
  }

  public equals( other: Trail ): boolean {
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
   */
  public upToNode( node: Node ): Trail {
    const nodeIndex = _.indexOf( this.nodes, node );
    assert && assert( nodeIndex >= 0, 'Trail does not contain the node' );
    return this.slice( 0, _.indexOf( this.nodes, node ) + 1 );
  }

  /**
   * Whether this trail contains the complete 'other' trail, but with added descendants afterwards.
   *
   * @param other - is other a subset of this trail?
   * @param allowSameTrail
   */
  public isExtensionOf( other: Trail, allowSameTrail?: boolean ): boolean {
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
   */
  public containsNode( node: Node ): boolean {
    return _.includes( this.nodes, node );
  }

  /**
   * A transform from our local coordinate frame to the other trail's local coordinate frame
   */
  public getTransformTo( otherTrail: Trail ): Transform3 {
    return new Transform3( this.getMatrixTo( otherTrail ) );
  }

  /**
   * Returns a matrix that transforms a point in our last node's local coordinate frame to the other trail's last node's
   * local coordinate frame
   */
  public getMatrixTo( otherTrail: Trail ): Matrix3 {
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
   *
   * If the trails are identical, the index should be equal to the trail's length.
   */
  public getBranchIndexTo( otherTrail: Trail ): number {
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
   */
  public getLastInputEnabledIndex(): number {
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
   */
  public getCursorCheckIndex(): number {
    return this.getLastInputEnabledIndex();
  }

  /**
   * TODO: phase out in favor of get()
   */
  public nodeFromTop( offset: number ): Node {
    return this.nodes[ this.length - 1 - offset ];
  }

  public lastNode(): Node {
    return this.nodeFromTop( 0 );
  }

  public rootNode(): Node {
    return this.nodes[ 0 ];
  }

  /**
   * Returns the previous graph trail in the order of self-rendering
   */
  public previous(): Trail | null {
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
   */
  public previousPainted(): Trail | null {
    let result = this.previous();
    while ( result && !result.isPainted() ) {
      result = result.previous();
    }
    return result;
  }

  /**
   * In the order of self-rendering
   */
  public next(): Trail | null {
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
   */
  public nextPainted(): Trail | null {
    let result = this.next();
    while ( result && !result.isPainted() ) {
      result = result.next();
    }
    return result;
  }

  /**
   * Calls callback( trail ) for this trail, and each descendant trail. If callback returns true, subtree will be skipped
   */
  public eachTrailUnder( callback: TrailCallback ): void {
    // TODO: performance: should be optimized to be much faster, since we don't have to deal with the before/after
    new TrailPointer( this, true ).eachTrailBetween( new TrailPointer( this, false ), callback );
  }

  /**
   * Standard Java-style compare. -1 means this trail is before (under) the other trail, 0 means equal, and 1 means this trail is
   * after (on top of) the other trail.
   * A shorter subtrail will compare as -1.
   *
   * Assumes that the Trails are properly indexed. If not, please reindex them!
   *
   * Comparison is for the rendering order, so an ancestor is 'before' a descendant
   */
  public compare( other: Trail ): number {
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

  public isBefore( other: Trail ): boolean {
    return this.compare( other ) === -1;
  }

  public isAfter( other: Trail ): boolean {
    return this.compare( other ) === 1;
  }

  public localToGlobalPoint( point: Vector2 ): Vector2 {
    // TODO: performance: multiple timesVector2 calls up the chain is probably faster
    return this.getMatrix().timesVector2( point );
  }

  public localToGlobalBounds( bounds: Bounds2 ): Bounds2 {
    return bounds.transformed( this.getMatrix() );
  }

  public globalToLocalPoint( point: Vector2 ): Vector2 {
    return this.getTransform().inversePosition2( point );
  }

  public globalToLocalBounds( bounds: Bounds2 ): Bounds2 {
    return this.getTransform().inverseBounds2( bounds );
  }

  public parentToGlobalPoint( point: Vector2 ): Vector2 {
    // TODO: performance: multiple timesVector2 calls up the chain is probably faster
    return this.getParentMatrix().timesVector2( point );
  }

  public parentToGlobalBounds( bounds: Bounds2 ): Bounds2 {
    return bounds.transformed( this.getParentMatrix() );
  }

  public globalToParentPoint( point: Vector2 ): Vector2 {
    return this.getParentTransform().inversePosition2( point );
  }

  public globalToParentBounds( bounds: Bounds2 ): Bounds2 {
    return this.getParentTransform().inverseBounds2( bounds );
  }

  private updateUniqueId(): void {
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
   */
  public getUniqueId(): string {
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
   */
  public toString(): string {
    this.reindex();
    if ( !this.length ) {
      return 'Empty Trail';
    }
    return `[Trail ${this.indices.join( '.' )} ${this.getUniqueId()}]`;
  }

  /**
   * Cleaner string form which will show class names. Not optimized by any means, meant for debugging.
   */
  public toPathString(): string {
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
   */
  public toDebugString(): string {
    return `${this.toString()} ${this.toPathString()}`;
  }

  /**
   * Like eachTrailBetween, but only fires for painted trails. If callback returns true, subtree will be skipped
   */
  public static eachPaintedTrailBetween( a: Trail, b: Trail, callback: ( trail: Trail ) => void, excludeEndTrails: boolean, rootNode: Node ): void {
    Trail.eachTrailBetween( a, b, ( trail: Trail ) => {
      if ( trail.isPainted() ) {
        return callback( trail );
      }
      return false;
    }, excludeEndTrails, rootNode );
  }

  /**
   * Global way of iterating across trails. when callback returns true, subtree will be skipped
   */
  public static eachTrailBetween( a: Trail, b: Trail, callback: ( trail: Trail ) => void, excludeEndTrails: boolean, rootNode: Node ): void {
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
   */
  public static branchIndex( a: Trail, b: Trail ): number {
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
   */
  public static sharedTrail( a: Trail, b: Trail ): Trail {
    return a.slice( 0, Trail.branchIndex( a, b ) );
  }

  /**
   * @param trailResults - Will be muted by appending matching trails
   * @param trail
   * @param predicate
   */
  public static appendAncestorTrailsWithPredicate( trailResults: Trail[], trail: Trail, predicate: ( node: Node ) => boolean ): void {
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
   * @param trailResults - Will be muted by appending matching trails
   * @param trail
   * @param predicate
   */
  public static appendDescendantTrailsWithPredicate( trailResults: Trail[], trail: Trail, predicate: ( node: Node ) => boolean ): void {
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
   */
  public static spannedSubtrees( a: Trail, b: Trail ): void {
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
   *
   * @param rootNode - the root of the trail being created
   * @param uniqueId - integers separated by ID_SEPARATOR, see getUniqueId
   */
  public static fromUniqueId( rootNode: Node, uniqueId: string ): Trail {
    const trailIds = uniqueId.split( ID_SEPARATOR );
    const trailIdNumbers = trailIds.map( id => Number( id ) );

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
        if ( children[ j ] !== null && children[ j ]!.id === trailId ) {
          const childAlongTrail = children[ j ]!;
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
